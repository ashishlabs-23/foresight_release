const API_BASE = window.FORESIGHT_API_BASE || "http://localhost:8000/api/v1";
const CARD_RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K'];
let state={sessionId:null,playerCards:[],dealerCards:[],config:{decks:6,hitSoft17:false,doubleAfterSplit:true,surrender:'late'},hands:{},activeHandId:null,submitting:false,soundEnabled:true,feltThemeIndex:0};
const FELT_THEMES = ['theme-emerald', 'theme-sapphire', 'theme-crimson'];

const $=id=>document.getElementById(id);
function showToast(msg,success=false){const el=success?$('success-banner'):$('error-banner');el.textContent=msg;el.classList.remove('hidden');setTimeout(()=>el.classList.add('hidden'),4500)}
function setLoading(on){$('ai-loading').classList.toggle('hidden',!on)}
function escapeHtml(v){return String(v).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
async function api(path,options={}){const res=await fetch(`${API_BASE}${path}`,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});let data=null;try{data=await res.json()}catch{}if(!res.ok)throw new Error(data?.detail||`Request failed (${res.status})`);return data}

// --- Synthesized Web Audio Sound Effects ---
let audioCtx = null;
function getAudioContext(){
  if(!audioCtx){
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if(AudioContextClass) audioCtx = new AudioContextClass();
  }
  if(audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

function playClickSound(){
  if(!state.soundEnabled) return;
  const ctx = getAudioContext();
  if(!ctx) return;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(440, ctx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.05);
  gain.gain.setValueAtTime(0.15, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.05);
  osc.connect(gain); gain.connect(ctx.destination);
  osc.start(); osc.stop(ctx.currentTime + 0.05);
}

function playDealSound(){
  if(!state.soundEnabled) return;
  const ctx = getAudioContext();
  if(!ctx) return;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'triangle';
  osc.frequency.setValueAtTime(220, ctx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(660, ctx.currentTime + 0.08);
  gain.gain.setValueAtTime(0.2, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);
  osc.connect(gain); gain.connect(ctx.destination);
  osc.start(); osc.stop(ctx.currentTime + 0.08);
}

function playWinSound(){
  if(!state.soundEnabled) return;
  const ctx = getAudioContext();
  if(!ctx) return;
  const now = ctx.currentTime;
  const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
  notes.forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, now + i * 0.1);
    gain.gain.setValueAtTime(0.2, now + i * 0.1);
    gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.1 + 0.3);
    osc.connect(gain); gain.connect(ctx.destination);
    osc.start(now + i * 0.1);
    osc.stop(now + i * 0.1 + 0.3);
  });
}

window.toggleSound = () => {
  state.soundEnabled = !state.soundEnabled;
  $('btn-sound').textContent = state.soundEnabled ? '🔊 Sound ON' : '🔇 Mute';
  showToast(state.soundEnabled ? 'Audio effects enabled' : 'Audio muted', true);
};

window.toggleFeltTheme = () => {
  state.feltThemeIndex = (state.feltThemeIndex + 1) % FELT_THEMES.length;
  document.body.className = FELT_THEMES[state.feltThemeIndex];
  const name = FELT_THEMES[state.feltThemeIndex].replace('theme-', '').toUpperCase();
  showToast(`Felt Theme: ${name}`, true);
};

function updateWorkflowStep(stepNum){
  for(let i = 1; i <= 4; i++){
    const el = $(`step-${i}`);
    if(!el) continue;
    el.classList.toggle('active', i === stepNum);
    el.classList.toggle('completed', i < stepNum);
  }
}

function getCardSuitSymbol(rank, index=0) {
  const suits = ['♠', '♥', '♦', '♣'];
  const code = (rank.charCodeAt(0) + index) % 4;
  return { symbol: suits[code], isRed: code === 1 || code === 2 };
}

function formatCardHTML(c, index=0) {
  const s = getCardSuitSymbol(c, index);
  return `<div class="card ${s.isRed ? 'red-suit' : 'black-suit'}">
    <span class="card-rank">${c}</span>
    <span class="card-suit">${s.symbol}</span>
  </div>`;
}

function formatMiniCardHTML(c, index=0) {
  const s = getCardSuitSymbol(c, index);
  return `<div class="card mini ${s.isRed ? 'red-suit' : 'black-suit'}">
    <span class="card-rank">${c}</span>
    <span class="card-suit">${s.symbol}</span>
  </div>`;
}

async function init(){
  renderSelectors();
  bindEvents();
  bindKeyboardHotkeys();
  await checkBackend();
  await fetchHistory();
  renderInputCards();
  initThreeBackground();
  updateWorkflowStep(1);
}

async function checkBackend(){
  const indicator=document.querySelector('.status-indicator');
  try{
    await api('/health',{headers:{}});
    indicator.innerHTML='<span class="dot connected"></span><span>Backend connected</span>';
  }catch{
    indicator.innerHTML='<span class="dot disconnected"></span><span>Backend offline</span>';
  }
}

function renderSelectors(){
  for(const id of ['selector-player','selector-dealer','selector-next','selector-dealer-final']){
    const el=$(id);
    if(!el)continue;
    el.innerHTML=CARD_RANKS.map(r=>`<button type="button" onclick="addCardTo('${r}','${id.includes('player')?'player':id.includes('dealer-final')?'dealer_final':id.includes('dealer')?'dealer':'next'}')">${r}</button>`).join('');
  }
}

function renderInputCards(){
  const p=$('input-player-cards'),d=$('input-dealer-cards');
  p.classList.toggle('empty',!state.playerCards.length);
  d.classList.toggle('empty',!state.dealerCards.length);
  p.innerHTML=state.playerCards.length?state.playerCards.map((c,i)=>formatCardHTML(c,i)).join(''):'<span>Select two cards below</span>';
  d.innerHTML=state.dealerCards.length?state.dealerCards.map((c,i)=>formatCardHTML(c,i+2)).join(''):'<span>Select a card below</span>';
}

window.addCardTo=(rank,target)=>{
  playDealSound();
  if(target==='player'){
    if(state.playerCards.length>=2)return showToast('Your opening hand already has two cards. Clear it to start over.');
    state.playerCards.push(rank);
    renderInputCards();
  }else if(target==='dealer'){
    if(state.dealerCards.length>=1)return showToast('Only one dealer up-card is needed to start analysis.');
    state.dealerCards.push(rank);
    renderInputCards();
  }else if(target==='next'){
    submitNextCard(rank,'player');
  }else if(target==='dealer_final'){
    submitNextCard(rank,'dealer');
  }
};

window.clearPlayer=()=>{playClickSound();state.playerCards=[];renderInputCards()};
window.clearDealer=()=>{playClickSound();state.dealerCards=[];renderInputCards()};

function bindEvents(){
  $('config-decks').onchange=e=>state.config.decks=+e.target.value;
  $('config-s17').onchange=e=>state.config.hitSoft17=e.target.value==='hit';
  $('config-das').onchange=e=>state.config.doubleAfterSplit=e.target.value==='true';
  $('config-surrender').onchange=e=>state.config.surrender=e.target.value;
  $('btn-analyze').onclick=analyzeState;
  $('btn-reset').onclick=resetSession;
  $('btn-submit-results').onclick=submitFinalResults;
}

function bindKeyboardHotkeys(){
  document.addEventListener('keydown', e => {
    if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)) return;
    const key = e.key.toUpperCase();
    if(key === 'ESCAPE'){ closeDetail(); closeSessionHelp(); }
    else if(key === 'N'){ resetSession(); }
    else if(state.sessionId && state.activeHandId){
      if(key === 'H') submitUserAction('hit');
      else if(key === 'S') submitUserAction('stand');
      else if(key === 'D') submitUserAction('double');
      else if(key === 'R') submitUserAction('surrender');
    }
  });
}

async function analyzeState(){
  playClickSound();
  if(state.playerCards.length!==2)return showToast('Enter exactly two player cards.');
  if(state.dealerCards.length!==1)return showToast('Enter one dealer up-card.');
  setLoading(true);
  $('empty-state').classList.add('hidden');
  const rules={decks:state.config.decks,hit_soft_17:state.config.hitSoft17,double_after_split:state.config.doubleAfterSplit,split_allowed:true,resplit_allowed:true,surrender_allowed:state.config.surrender!=='none',late_surrender:state.config.surrender==='late',early_surrender:state.config.surrender==='early'};
  const hand_id='h1_'+Date.now();
  try{
    const data=await api('/analyzer/analyze',{method:'POST',body:JSON.stringify({rules,player_hands:[{hand_id,cards:state.playerCards,is_active:true,is_completed:false}],dealer_cards:state.dealerCards})});
    updateSessionState(data);
    updateWorkflowStep(2);
    $('session-view').classList.remove('hidden');
    $('session-id').textContent=data.session_id;
    $('session-chip').textContent='ACTIVE · '+data.session_id.slice(0,8);
    showToast('Analysis ready. Choose your action [H, S, D, R].',true);
  }catch(e){
    showToast(e.message);
  }finally{
    setLoading(false);
  }
}

function updateSessionState(data){
  state.sessionId=data.session_id||state.sessionId;
  state.hands=data.hands||{};
  state.activeHandId=data.active_hand_id||null;
  if(data.dealer_cards)state.dealerCards=data.dealer_cards;
  renderSessionView();
  
  if(state.activeHandId&&data.recommendation&&data.recommendation.status!=='all_completed'){
    renderRecommendation(data.recommendation);
    updateWorkflowStep(3);
    $('active-hand-panel').classList.remove('hidden');
    $('next-card-section').classList.add('hidden');
    $('resolution-section').classList.add('hidden');
  }else if(state.activeHandId){
    $('active-hand-panel').classList.add('hidden');
    $('next-card-section').classList.remove('hidden');
    $('resolution-section').classList.add('hidden');
  }else{
    finishSessionView();
  }
}

function renderSessionView(){
  $('session-dealer-cards').innerHTML=state.dealerCards.map((c,i)=>formatCardHTML(c,i+2)).join('');
  $('player-hands-container').innerHTML=Object.entries(state.hands).map(([hid,h],i)=>`<div class="hand-box ${hid===state.activeHandId?'active':''}"><h4>HAND ${i+1} ${hid===state.activeHandId?'· ACTIVE':''}</h4><div class="hand-cards">${h.cards.map((c,j)=>formatCardHTML(c,j)).join('')}</div></div>`).join('');
}

function renderRecommendation(rec){
  $('rec-action').textContent=(rec.recommended_action||'—').toUpperCase();
  $('rec-reason').textContent=rec.reason||'';
  $('m-margin').textContent=Number(rec.decision_margin||0).toFixed(3);
  $('m-uncertainty').textContent=(Number(rec.uncertainty||0)*100).toFixed(1)+'%';
  $('m-risk').textContent=(rec.risk_level||'—').toUpperCase();
  $('support-badge').textContent=(rec.support_level||'EXACT').toUpperCase();
  
  const values=rec.action_analysis||{};
  const max=Math.max(...Object.values(values).map(Number),0.001);
  $('ev-list').innerHTML=Object.entries(values).sort((a,b)=>b[1]-a[1]).map(([a,v])=>`<li class="ev-row"><span>${a.toUpperCase()}</span><div class="ev-bar"><i style="width:${Math.max(4,Math.min(100,Math.abs(v)/max*100))}%"></i></div><b>${Number(v).toFixed(3)}</b></li>`).join('');
  
  const hotkeyMap = { 'hit': 'H', 'stand': 'S', 'double': 'D', 'surrender': 'R', 'split': 'P' };
  $('action-controls').innerHTML=Object.keys(values).map(a=>`<button type="button" class="action-btn ${a===rec.recommended_action?'recommended':''}" onclick="submitUserAction('${a}')">${a.toUpperCase()} <kbd>${hotkeyMap[a]||''}</kbd></button>`).join('');
  $('feedback-state').textContent='Awaiting your action';
  $('feedback-quality').textContent='Ready';
}

window.submitUserAction=async action=>{
  if(state.submitting||!state.sessionId||!state.activeHandId)return;
  playClickSound();
  state.submitting=true;
  document.querySelectorAll('.action-btn').forEach(b=>b.disabled=true);
  $('feedback-state').textContent='Recording…';
  try{
    const data=await api('/analyzer/decision/action',{method:'POST',body:JSON.stringify({session_id:state.sessionId,hand_id:state.activeHandId,user_action:action})});
    $('feedback-state').textContent='Recorded ✓';
    updateSessionState(data);
    if((action==='hit'||action==='double')&&data.active_hand_id){
      $('next-card-section').classList.remove('hidden');
      showToast(`${action.toUpperCase()} recorded. Enter dealt card.`,true);
    }else if(!data.active_hand_id){
      showToast('All player hands completed. Enter dealer cards below.',true);
    }
  }catch(e){
    $('feedback-state').textContent='Try again';
    showToast(e.message);
  }finally{
    state.submitting=false;
    document.querySelectorAll('.action-btn').forEach(b=>b.disabled=false);
  }
};

async function submitNextCard(rank,target){
  if(state.submitting||!state.sessionId)return;
  playDealSound();
  state.submitting=true;
  setLoading(true);
  try{
    const body={session_id:state.sessionId,target,card:rank};
    if(target==='player')body.hand_id=state.activeHandId;
    const data=await api('/analyzer/decision/card',{method:'POST',body:JSON.stringify(body)});
    updateSessionState(data);
    showToast(`Card ${rank} recorded.`,true);
  }catch(e){
    showToast(e.message);
  }finally{
    state.submitting=false;
    setLoading(false);
  }
}

function calculateHandValue(cards){
  let total=0, aces=0;
  for(const c of cards){
    if(c==='A'){ aces++; total+=11; }
    else if(['K','Q','J','10'].includes(c)){ total+=10; }
    else { total+=parseInt(c,10)||10; }
  }
  while(total>21 && aces>0){ total-=10; aces--; }
  return total;
}

function calculateOutcome(playerCards, dealerCards){
  const pVal = calculateHandValue(playerCards);
  const dVal = calculateHandValue(dealerCards);
  
  if(pVal > 21) return 'bust';
  if(playerCards.length === 2 && pVal === 21 && (dealerCards.length !== 2 || dVal !== 21)) return 'blackjack';
  if(dVal > 21) return 'win';
  if(pVal > dVal) return 'win';
  if(pVal < dVal) return 'loss';
  return 'push';
}

function finishSessionView(){
  updateWorkflowStep(4);
  $('active-hand-panel').classList.add('hidden');
  $('next-card-section').classList.add('hidden');
  $('resolution-section').classList.remove('hidden');
  renderOutcomeSelectors();
  $('feedback-quality').textContent='Ready to close';
}

function renderDealerFinalDisplay(){
  const dEl = $('dealer-cards-display');
  const tEl = $('dealer-total-badge');
  if(dEl){
    dEl.innerHTML = state.dealerCards.map((c,i) => formatCardHTML(c,i+2)).join('');
  }
  if(tEl){
    const total = calculateHandValue(state.dealerCards);
    tEl.textContent = state.dealerCards.length >= 2 ? `Total: ${total}${total > 21 ? ' (DEALER BUST)' : ''}` : `Total: ${total} (Awaiting Hole Card)`;
  }
}

function renderOutcomeSelectors(){
  renderDealerFinalDisplay();
  const container = $('final-results-container');
  if(!container) return;
  
  container.innerHTML = Object.entries(state.hands).map(([hid, h], i) => {
    const pTotal = calculateHandValue(h.cards);
    const autoRes = state.dealerCards.length >= 2 ? calculateOutcome(h.cards, state.dealerCards) : 'win';
    
    return `<div class="result-row">
      <div style="display:flex;align-items:center;gap:10px;">
        <b style="color:var(--blue);">Hand ${i+1}</b>
        <div style="display:flex;gap:6px;">${h.cards.map((c,j)=>formatCardHTML(c,j)).join('')}</div>
        <span style="color:#8fa4bd;margin-left:8px;font-size:11px;">(Total: <b>${pTotal}</b>${pTotal>21?' BUST':''})</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:10px;color:#687484;">Outcome:</span>
        <select id="res-${hid}">
          <option value="win" ${autoRes==='win'?'selected':''}>Win</option>
          <option value="loss" ${autoRes==='loss'?'selected':''}>Loss</option>
          <option value="push" ${autoRes==='push'?'selected':''}>Push</option>
          <option value="blackjack" ${autoRes==='blackjack'?'selected':''}>Blackjack</option>
          <option value="bust" ${autoRes==='bust'?'selected':''}>Bust</option>
        </select>
      </div>
    </div>`;
  }).join('');
}

function triggerVictoryConfetti(){
  playWinSound();
  const canvas = $('confetti-canvas');
  if(!canvas) return;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext('2d');
  
  const particles = Array.from({length: 80}, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height - canvas.height,
    r: Math.random() * 6 + 4,
    d: Math.random() * 80,
    color: ['#10b981', '#38bdf8', '#fbbf24', '#f43f5e', '#a855f7'][Math.floor(Math.random()*5)],
    tilt: Math.random() * 10 - 10,
    tiltAngleIncremental: Math.random() * 0.07 + 0.05,
    tiltAngle: 0
  }));

  let animationFrame;
  let counter = 0;
  function draw(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.tiltAngle += p.tiltAngleIncremental;
      p.y += (Math.cos(p.d) + 3 + p.r / 2) / 2;
      p.tilt = Math.sin(p.tiltAngle) * 15;
      ctx.beginPath();
      ctx.lineWidth = p.r;
      ctx.strokeStyle = p.color;
      ctx.moveTo(p.x + p.tilt + p.r / 2, p.y);
      ctx.lineTo(p.x + p.tilt, p.y + p.tilt + p.r / 2);
      ctx.stroke();
    });
    counter++;
    if(counter < 180) animationFrame = requestAnimationFrame(draw);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  draw();
}

async function submitFinalResults(){
  if(state.submitting||!state.sessionId)return;
  const results={};
  let hasWin = false;
  for(const hid of Object.keys(state.hands)){
    const el=$(`res-${hid}`);
    if(!el)return showToast('Could not read the result for one of the hands.');
    results[hid]=el.value;
    if(el.value === 'win' || el.value === 'blackjack') hasWin = true;
  }
  state.submitting=true;
  $('btn-submit-results').disabled=true;
  $('btn-submit-results').textContent='Closing session…';
  try{
    const data=await api('/analyzer/decision/result',{method:'POST',body:JSON.stringify({session_id:state.sessionId,hand_results:results})});
    if(data.status!=='success'||data.closed!==true)throw new Error('The server did not confirm session closure.');
    if(hasWin) triggerVictoryConfetti();
    showToast('Session closed and feedback saved.',true);
    resetSession(false);
    await fetchHistory();
  }catch(e){
    showToast('Session was not closed: '+e.message);
  }finally{
    state.submitting=false;
    $('btn-submit-results').disabled=false;
    $('btn-submit-results').textContent='Submit results & close session';
  }
}

function resetSession(showMessage=true){
  playClickSound();
  state.sessionId=null;
  state.playerCards=[];
  state.dealerCards=[];
  state.hands={};
  state.activeHandId=null;
  state.submitting=false;
  renderInputCards();
  updateWorkflowStep(1);
  $('session-view').classList.add('hidden');
  $('empty-state').classList.remove('hidden');
  $('session-id').textContent='—';
  $('session-chip').textContent='SESSION';
  if(showMessage)showToast('New session ready.',true);
}

async function fetchHistory(){
  try{
    const history=await api('/analyzer/decision/history');
    $('history-list').innerHTML=history.length?history.slice(0,15).map(h=>{
      const snapshot = h.state_snapshot || {};
      const pCards = [...(snapshot.player_cards||[]), ...(h.new_cards||[])];
      const dCards = snapshot.dealer_cards || [];
      const resStr = (h.final_result || 'Pending').toLowerCase();
      const resColor = resStr === 'win' ? 'var(--green)' : resStr === 'loss' ? 'var(--red)' : resStr === 'blackjack' ? 'var(--gold)' : 'var(--blue)';
      
      return `<li class="history-item" onclick="showDetail('${escapeHtml(h.decision_id)}')">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <small style="color:#64748b;font-weight:700;">${new Date(h.timestamp).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</small>
          <span style="font-size:9px;font-weight:800;color:${resColor};border:1px solid ${resColor}44;background:${resColor}15;padding:2px 7px;border-radius:99px;">${escapeHtml((h.final_result||'Pending').toUpperCase())}</span>
        </div>
        <b style="margin:2px 0 6px;color:#e2e8f0;font-size:11px;display:block;">REC: <span style="color:var(--green);">${(h.recommended_action||'').toUpperCase()}</span> · USER: <span style="color:var(--blue);">${(h.user_action||'—').toUpperCase()}</span></b>
        <div style="display:grid;gap:4px;margin-top:4px;background:rgba(0,0,0,0.25);padding:6px;border-radius:8px;border:1px solid rgba(255,255,255,0.05);">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:9px;color:#7d8796;font-weight:800;min-width:38px;">PLAYER</span>
            <div style="display:flex;gap:3px;flex-wrap:wrap;">${pCards.length ? pCards.map((c,i)=>formatMiniCardHTML(c,i)).join('') : '—'}</div>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:9px;color:#7d8796;font-weight:800;min-width:38px;">DEALER</span>
            <div style="display:flex;gap:3px;flex-wrap:wrap;">${dCards.length ? dCards.map((c,i)=>formatMiniCardHTML(c,i+2)).join('') : '—'}</div>
          </div>
        </div>
      </li>`;
    }).join(''):'<li class="history-empty">No decisions yet.</li>';
  }catch(e){
    console.warn(e);
  }
}

window.showDetail=async id=>{
  playClickSound();
  try{
    const data = await api('/analyzer/decision/'+encodeURIComponent(id));
    const snapshot = data.state_snapshot || {};
    const playerBaseCards = snapshot.player_cards || [];
    const newCards = data.new_cards || [];
    const playerFullCards = [...playerBaseCards, ...newCards];
    const dealerFullCards = snapshot.dealer_cards || [];

    const dCardsEl = $('detail-dealer-cards');
    const pCardsEl = $('detail-player-cards');
    if (dCardsEl) dCardsEl.innerHTML = dealerFullCards.length ? dealerFullCards.map((c,i) => formatCardHTML(c,i+2)).join('') : '<div class="card">—</div>';
    if (pCardsEl) pCardsEl.innerHTML = playerFullCards.length ? playerFullCards.map((c,j) => formatCardHTML(c,j)).join('') : '<div class="card">—</div>';

    const pVal = calculateHandValue(playerFullCards);
    const dVal = calculateHandValue(dealerFullCards);
    if($('detail-player-total')) $('detail-player-total').textContent = `Total: ${pVal}${pVal > 21 ? ' (BUST)' : ''}`;
    if($('detail-dealer-total')) $('detail-dealer-total').textContent = `Total: ${dVal}${dVal > 21 ? ' (DEALER BUST)' : ''}`;

    if($('detail-rec-action')) $('detail-rec-action').textContent = (data.recommended_action || '—').toUpperCase();
    if($('detail-user-action')) $('detail-user-action').textContent = (data.user_action || '—').toUpperCase();
    if($('detail-final-result')) $('detail-final-result').textContent = (data.final_result || 'Pending Result').toUpperCase();

    const values = data.predicted_evs || {};
    const max = Math.max(...Object.values(values).map(Number), 0.001);
    const evListEl = $('detail-ev-list');
    if (evListEl) {
      evListEl.innerHTML = Object.entries(values).sort((a, b) => b[1] - a[1]).map(([a, v]) => 
        `<li class="ev-row"><span>${a.toUpperCase()}</span><div class="ev-bar"><i style="width:${Math.max(4, Math.min(100, Math.abs(v) / max * 100))}%"></i></div><b>${Number(v).toFixed(3)}</b></li>`
      ).join('');
    }

    $('detail-content').textContent = JSON.stringify(data, null, 2);
    $('detail-modal').classList.remove('hidden');
  }catch(e){
    showToast(e.message);
  }
};

window.closeDetail=()=>$('detail-modal').classList.add('hidden');
window.openSessionHelp=()=>$('help-modal').classList.remove('hidden');
window.closeSessionHelp=()=>$('help-modal').classList.add('hidden');

function initThreeBackground() {
  if (typeof THREE === 'undefined') return;
  try {
    const canvas = document.createElement('canvas');
    canvas.id = 'three-bg-canvas';
    canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;opacity:0.65;';
    document.body.prepend(canvas);

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 300;

    const count = 75;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const speeds = [];

    const palette = [
      new THREE.Color('#10b981'),
      new THREE.Color('#38bdf8'),
      new THREE.Color('#fbbf24'),
      new THREE.Color('#a855f7')
    ];

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 900;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 700;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 500;

      const col = palette[i % palette.length];
      colors[i * 3] = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;

      speeds.push({
        x: (Math.random() - 0.5) * 0.25,
        y: Math.random() * 0.35 + 0.15
      });
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 6,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending
    });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    function animate() {
      requestAnimationFrame(animate);
      const pos = geometry.attributes.position.array;
      for (let i = 0; i < count; i++) {
        pos[i * 3 + 1] += speeds[i].y;
        pos[i * 3] += speeds[i].x;
        if (pos[i * 3 + 1] > 380) pos[i * 3 + 1] = -380;
        if (pos[i * 3] > 480) pos[i * 3] = -480;
        if (pos[i * 3] < -480) pos[i * 3] = 480;
      }
      geometry.attributes.position.needsUpdate = true;
      points.rotation.y += 0.0006;
      points.rotation.x += 0.0003;
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  } catch (e) {
    console.warn('Three.js background error:', e);
  }
}

document.addEventListener('DOMContentLoaded',init);
