# Frontend

> **Phase 5** — Not yet implemented.

The frontend will be a React/Next.js application providing:
- An interactive Blackjack table (play vs AI or watch AI play)
- Strategy heatmap visualization (Basic Strategy chart)
- Training curves and win-rate charts
- Real-time WebSocket game sessions

## Communication

The frontend communicates with the backend exclusively via:
- `REST API` — `http://localhost:8000/api/v1/`
- `WebSocket` — `ws://localhost:8000/ws/` (Phase 4+)

## Tech Stack (Planned)

| Concern | Technology |
|---------|-----------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Charts | Recharts / D3 |
| WebSocket | native browser WS |
| State | Zustand |
