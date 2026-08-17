"""Fast exact blackjack EV analysis using dynamic programming.

HIT/STAND/DOUBLE/SURRENDER are evaluated from the currently observed shoe
composition. No random simulation is used. SPLIT remains a separate rule-based
branch because exact multi-hand split EV requires shared-shoe state tracking.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple
from blackjack.strategies.base import Action
from blackjack.rules.rules import BlackjackRules

RANKS: Tuple[str, ...] = ("2","3","4","5","6","7","8","9","T","A")
VALUES = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"A":11}
IDX = {r:i for i,r in enumerate(RANKS)}

def norm_rank(r:str)->str:
    r=r.upper()
    return "T" if r in {"10","J","Q","K"} else r

def total_soft(cards:Tuple[str,...])->tuple[int,bool]:
    hard=sum(1 if c=="A" else VALUES[c] for c in cards)
    aces=cards.count("A")
    total=hard+10 if aces and hard+10<=21 else hard
    soft=aces>0 and total==hard+10 and total<21
    return total,soft

def add_card_state(total:int, soft:bool, rank:str)->tuple[int,bool]:
    # Keep every new Ace at 1 first, then promote one Ace to 11 when possible.
    if rank == "A":
        t = total + 1
        if t + 10 <= 21:
            t += 10
            return t, True
        return t, False
    t = total + VALUES[rank]
    if t > 21 and soft:
        t -= 10
        soft = False
    return t, soft


@dataclass(frozen=True)
class ExactActionResult:
    action:str
    ev:float
    lower:float
    upper:float

class ExactEVAnalyzer:
    def __init__(self,rules:BlackjackRules): self.rules=rules

    def _counts(self,decks:int,observed:list[str])->tuple[int,...]:
        c=[4*decks]*10
        for raw in observed:
            r=norm_rank(raw)
            if r not in IDX: continue
            c[IDX[r]]-=1
        if min(c)<0: raise ValueError("Observed cards exceed configured shoe composition")
        return tuple(c)

    def evaluate(self,player_cards:list[str],dealer_upcard:str,observed_cards:list[str],decks:int,legal_actions:list[Action])->Dict[str,ExactActionResult]:
        p=tuple(norm_rank(x) for x in player_cards); d=norm_rank(dealer_upcard)
        counts=self._counts(decks,observed_cards)

        def draw_options(c):
            n=sum(c)
            for i,r in enumerate(RANKS):
                if c[i]:
                    nc=list(c); nc[i]-=1
                    yield r,c[i]/n,tuple(nc)

        # Dealer terminal distribution after the hole card is known.
        # Tuple: bust, 17,18,19,20,21, natural.
        @lru_cache(maxsize=300000)
        def dealer_dist(total:int,soft:bool,c:tuple[int,...],cards:int,natural_possible:bool)->tuple[float,...]:
            if total>21: return (1.,0.,0.,0.,0.,0.,0.)
            if natural_possible and cards==2 and total==21:
                return (0.,0.,0.,0.,0.,0.,1.)
            if not self.rules.dealer_must_hit(total,soft):
                out=[0.]*7; out[total-16]=1.; return tuple(out)
            out=[0.]*7
            for r,prob,nc in draw_options(c):
                nt,ns=add_card_state(total,soft,r)
                sub=dealer_dist(nt,ns,nc,cards+1,False)
                for i,v in enumerate(sub): out[i]+=prob*v
            return tuple(out)

        @lru_cache(maxsize=300000)
        def stand_ev(total:int,soft:bool,c:tuple[int,...])->float:
            # A busted player hand has no dealer contest.
            if total > 21:
                return -1.0
            # A natural blackjack is paid at the configured natural payout
            # unless the dealer also has a natural, in which case it pushes.
            is_player_natural = len(p) == 2 and total == 21
            natural_payout = 1.5 if self.rules.blackjack_payout.value == "3:2" else 1.2
            # Enumerate the hidden dealer hole card exactly.
            ev=0.0
            for r,prob,nc in draw_options(c):
                ht,hs=add_card_state(VALUES[d] if d!='A' else 11, d=='A', r)
                # Correct Ace adjustment for dealer upcard + hole card.
                if d=='A' and r=='A': ht,hs=12,True
                elif d=='A' and r!='A': ht,hs=11+VALUES[r],True
                if ht>21 and hs: ht-=10; hs=False
                elif d!='A' and r=='A':
                    ht=VALUES[d]+11
                    if ht>21: ht-=10; hs=False
                    else: hs=True
                dist=dealer_dist(ht,hs,nc,2,True)
                if is_player_natural:
                    # Dealer natural pushes; otherwise the player's natural is paid.
                    sub = (1.0 - dist[6]) * natural_payout
                else:
                    sub=-dist[6]  # natural dealer blackjack
                    for i,pr in enumerate(dist[1:6],start=1):
                        dt=16+i
                        if total>dt: sub+=pr
                        elif total<dt: sub-=pr
                    sub+=dist[0]
                ev+=prob*sub
            return ev

        @lru_cache(maxsize=300000)
        def best_after_hit(total:int,soft:bool,c:tuple[int,...])->float:
            if total>21:return -1.0
            if total==21:return stand_ev(total,soft,c)
            # After a hit, double is no longer legal.
            hit=sum(prob*best_after_hit(*add_card_state(total,soft,r),nc) for r,prob,nc in draw_options(c))
            return max(stand_ev(total,soft,c),hit)

        ptotal,psoft=total_soft(p)
        results={}
        for action in legal_actions:
            if action==Action.STAND:
                ev=stand_ev(ptotal,psoft,counts)
            elif action==Action.HIT:
                ev=sum(prob*best_after_hit(*add_card_state(ptotal,psoft,r),nc) for r,prob,nc in draw_options(counts))
            elif action==Action.DOUBLE:
                ev=2.0*sum(prob*stand_ev(*add_card_state(ptotal,psoft,r),nc) for r,prob,nc in draw_options(counts))
            elif action==Action.SURRENDER:
                ev=-0.5
            else: continue
            results[action.name.lower()]=ExactActionResult(action.name.lower(),float(ev),float(ev),float(ev))
        return results
