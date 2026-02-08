# Hierarchical DRL Multi-Strategy Fund - 20 Slide Outline

Slide 1 - Title
- Hierarchical DRL Multi-Strategy Fund
- Adaptive capital allocation across 7 specialists + CIO allocator
- Presenter: <Your Name> | Stony Brook Math & Data Science Conference | Feb 2026

Slide 2 - Agenda
- Why this project exists
- System architecture and data scope
- Specialist agents (math, strategy, market dependence)
- Training + decision processes
- Results and future improvements

Slide 3 - Reason for the Project
- Markets shift regimes; single strategies overfit to one regime
- Need modular experts with a capital allocator that adapts
- Goal: maximize risk-adjusted return under realistic costs and constraints
- Research focus: hierarchical RL as a portfolio decision system

Slide 4 - Problem Framing (Hierarchical MDP)
- Specialists solve $\max_\pi E[\sum_t \gamma^t r_t^i]$ in their market microstructure
- CIO chooses weights $w_t \in \Delta^7$ to maximize $E[\sum_t \gamma^t (r_t - \lambda \sigma_t^2)]$
- Two-level control: $a_t^i$ at strategy level, $w_t$ at fund level

Slide 5 - Data and Markets
- Assets: Equities (pairs), FX, Futures, Options (greeks)
- Data source: ArcticDB with technical + microstructure features
- Splits: Train 2010-2018, Val 2019, Test 2020-2024
- Costs and constraints modeled per strategy

Slide 6 - System Architecture (High Level)
- Feature engineering -> 7 specialist environments -> master CIO allocator
- Specialists output strategy PnL; CIO sets allocation weights
- Backtesting engine enforces costs, leverage, and turnover limits

Slide 7 - Master CIO Allocator (PPO)
- Math: PPO objective $L^{CLIP}(\theta)=E[\min(r_t(\theta)\hat{A}_t,\,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]$
- Strategy: continuous allocation weights to balance return vs risk
- Market effect: regime shifts change specialist correlations and vol
- Diagram: drawio/cio_allocator.drawio

Slide 8 - Statistical Arbitrage (PPO)
- Math: mean-reversion spread trading with continuous position $a_t \in [-1,1]$
- Strategy: cointegrated pairs, z-score timing, hedge ratio control
- Market effect: regime breaks + volatility alter reversion speed
- Diagram: drawio/statistical_arbitrage.drawio

Slide 9 - Market Making (DDPG)
- Math: deterministic policy gradient $\nabla_\theta J \approx E[\nabla_a Q(s,a)|_{a=\mu_\theta(s)}\nabla_\theta \mu_\theta(s)]$
- Strategy: bid/ask offset control, inventory targeting, fill models
- Market effect: spread width + order flow shift optimal quotes
- Diagram: drawio/market_making.drawio

Slide 10 - Factor Tracking (DQN)
- Math: Bellman update $Q(s,a)=r+\gamma \max_{a'} Q(s',a')$
- Strategy: discrete factor tilts (long/neutral/short) across factors
- Market effect: factor momentum + correlation shifts change optimal tilt
- Diagram: drawio/factor_tracking.drawio

Slide 11 - Volatility Trading (PPO)
- Math: continuous vega exposure, reward from vol risk premium
- Strategy: delta-hedged vol positions, term-structure signals
- Market effect: IV-RV gap, vol-of-vol, and event risk drive actions
- Diagram: drawio/volatility_trading.drawio

Slide 12 - Delta Hedging (DDPG)
- Math: continuous hedge ratio control to minimize hedging error
- Strategy: manage Greeks (delta/gamma/vega) under transaction costs
- Market effect: jumps and smile dynamics change hedge frequency
- Diagram: drawio/delta_hedging.drawio

Slide 13 - Futures Spreads (PPO)
- Math: continuous spread position with stationarity bonuses
- Strategy: calendar/inter-commodity spreads, roll-yield signals
- Market effect: seasonality and carry change spread direction
- Diagram: drawio/futures_spreads.drawio

Slide 14 - FX Arbitrage (DDPG)
- Math: continuous multi-leg positions for triangular deviations
- Strategy: exploit mispricings + carry; manage funding costs
- Market effect: rate differentials + liquidity shocks affect sizing
- Diagram: drawio/fx_arbitrage.drawio

Slide 15 - Agent Process (Per-Step Loop)
- Observe state -> normalize features -> policy outputs action
- Environment simulates fills, costs, constraints -> reward
- PPO on-policy updates; DDPG/DQN replay buffer off-policy updates

Slide 16 - Training Pipeline
- Phase 1: Train specialists independently (500K-1M steps each)
- Phase 2: Train CIO allocator with pretrained specialists (200K steps)
- Phase 3: Optional joint fine-tuning with lower specialist LR

Slide 17 - Decision Process Differences
- Continuous vs discrete actions define algorithm choice
- Strategy-specific penalties: inventory risk, hedge error, turnover
- Market regime features gate risk-taking and position size

Slide 18 - Results (Master CIO)
- Sharpe 1.60, Max Drawdown -0.80%, Annual Return 7.20%
- 6x Sharpe vs equal-weight; 78% lower drawdown
- Plots: reports/plots/equity_curves_comparison.png

Slide 19 - Specialist Results (Highlights)
- Stat Arb: 25.22% return, Sharpe 0.58
- Market Making: 17.66% return, Sharpe 0.85
- CIO allocation reduces exposure to underperformers

Slide 20 - Future Improvements
- Regime-aware reward shaping and adversarial stress tests
- Transaction cost calibration to venue-specific data
- Multi-task training for transfer across strategies
- Closing: hierarchical RL as a scalable fund architecture
