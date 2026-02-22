# Hierarchical DRL Multi-Strategy Fund - 21 Slide Outline

Slide 1 - Title
- Hierarchical DRL Multi-Strategy Fund
- Adaptive capital allocation across 7 specialists + CIO allocator
- Presenter: <Your Name> | Math & Data Science Conference | Feb 2026

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

Slide 7 - RL Environment Framework (Gymnasium MDP)
- Math: Markov Decision Process formulation for financial trading
  - State space: $\mathcal{S} = \mathbb{R}^d$ where $s_t = [p_t, f_t, x_t]$ (prices, features, portfolio state)
  - Action space: $\mathcal{A} = \mathbb{R}^m$ (continuous) or $\mathcal{A} = \{0,1,...,n\}$ (discrete)
  - Transition: $s_{t+1} = T(s_t, a_t, \omega_t)$ where $\omega_t$ is market dynamics
  - Reward: $r_t = \Delta V_t - c_{\text{trans}} - \sum_i \lambda_i \cdot \text{Penalty}_i$ (P&L minus costs and penalties)
  - Value function: $V^\pi(s) = E[\sum_{k=0}^\infty \gamma^k r_{t+k} | s_t=s, \pi]$ with $\gamma \in [0.95, 0.99]$
  - Portfolio value: $V_t = \text{Cash}_t + \sum_i n_{i,t} P_{i,t}$ (cash + positions × prices)
  - Transaction cost: $c_t = \sum_i |\Delta n_i| P_i (c_{\%} + s_{\%})$ (cost \% + slippage \%)
  - Risk metrics: Sharpe $= \frac{\mu_r - r_f}{\sigma_r}$, Drawdown $= \max_\tau (V_\tau - V_t) / V_\tau$
- Implementation: Abstract `BaseTradingEnv(gym.Env)` with `step()`, `reset()`, `_take_action()`, `_calculate_reward()`
- Child classes define strategy-specific $\mathcal{S}$, $\mathcal{A}$, and $r_t$ components

Slide 8 - Master CIO Allocator (PPO)
- Math: Portfolio optimization with hierarchical control
  - PPO objective: $L^{CLIP}(\theta)=E[\min(r_t(\theta)\hat{A}_t,\,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]$
  - Weights: $w_t \in [0,1]^7$ with $\sum_i w_i = 1$ and $w_i \in [w_{\text{min}}, w_{\text{max}}]$
  - Portfolio return: $r_{p,t} = \sum_{i=1}^7 w_{i,t} \cdot r_{i,t}$ (specialist returns)
  - Turnover: $\tau_t = \sum_i |w_{i,t} - w_{i,t-1}|$
  - Diversification: $D = -\sum_i w_i \log(w_i) / \log(7)$ (normalized entropy)
  - Reward: $R_t = 100 r_{p,t} + 0.1 \cdot \text{Sharpe}_{60} + 0.05 D - \lambda_1 \tau_t c_{\text{trans}} - \lambda_2 \mathbb{1}_{\tau_t > \tau_{\text{max}}}$
- Strategy: continuous allocation weights to balance return vs risk, entropy-based diversification
- Market effect: regime shifts change specialist correlations and vol
- Diagram: drawio/cio_allocator.drawio

Slide 9 - Statistical Arbitrage (PPO)
- Math: Pairs trading with cointegration-based mean reversion
  - Spread: $S_t = P_{1,t} - \beta \cdot P_{2,t}$ where $\beta$ is hedge ratio
  - Z-score: $z_t = \frac{S_t - \mu_S}{\sigma_S}$ (rolling window)
  - Half-life: $\tau_{1/2} = -\ln(2) / \lambda$ from AR(1): $S_t = \alpha + \lambda S_{t-1} + \epsilon_t$
  - Position sizing: $a_t \in [-1, 1]$ scaled to $\$X$, giving shares: $n_1 = a_t X / P_1$, $n_2 = -a_t X \beta / P_2$
  - P&L: $\text{PnL}_t = n_1 \Delta P_1 + n_2 \Delta P_2 - c_{\text{trans}}(|\Delta n_1| P_1 + |\Delta n_2| P_2)$
  - Reward: $R_t = 1000 \cdot \frac{\Delta \text{PnL}}{V_0} + 0.5 \cdot \mathbb{1}_{|z_t|>2, \text{sign}(z_t \cdot a_t)<0} - 0.1 |a_t| \cdot \mathbb{1}_{|z_t|<0.5}$
- Strategy: cointegrated pairs, z-score timing, hedge ratio control, mean reversion entry/exit
- Market effect: regime breaks + volatility alter reversion speed
- Diagram: drawio/statistical_arbitrage.drawio

Slide 10 - Market Making (DDPG)
- Math: Optimal quote placement with inventory risk management
  - DDPG gradient: $\nabla_\theta J \approx E[\nabla_a Q(s,a)|_{a=\mu_\theta(s)}\nabla_\theta \mu_\theta(s)]$
  - Actions: $(\delta_{\text{bid}}, \delta_{\text{ask}}) \in [-1,1]^2$ scaled to $[s_{\text{min}}, s_{\text{max}}]$
  - Quotes: $P_{\text{bid}} = P_{\text{mid}}(1 - \delta_{\text{bid}})$, $P_{\text{ask}} = P_{\text{mid}}(1 + \delta_{\text{ask}})$
  - Fill probabilities: $p_{\text{fill}} = \max(0.1, 1 - \delta / s_{\text{max}}) \cdot 0.5$ (tighter spreads → higher fills)
  - Inventory: $I_t = I_{t-1} + n_{\text{bid}} - n_{\text{ask}}$ with $|I_t| \leq I_{\text{max}}$
  - P&L: $\text{PnL}_t = n_{\text{ask}} P_{\text{ask}} - n_{\text{bid}} P_{\text{bid}} + c_{\text{rebate}}(n_{\text{bid}} + n_{\text{ask}})$
  - Reward: $R_t = \Delta \text{PnL} - \lambda_1(I_t / I_{\text{max}})^2 + \lambda_2(f_{\text{bid}} + f_{\text{ask}}) + \lambda_3 \mathbb{1}_{\text{both filled}} - \lambda_4 \mathbb{1}_{\text{adverse}}$
- Strategy: bid/ask offset control, inventory targeting, fill models, maker rebates
- Market effect: spread width + order flow shift optimal quotes
- Diagram: drawio/market_making.drawio

Slide 11 - Factor Tracking (DQN)
- Math: Multi-factor portfolio construction with discrete tilts
  - Bellman optimality: $Q(s,a)=r+\gamma \max_{a'} Q(s',a')$
  - Action space: $a \in \{0,1,2\}^K$ for $K$ factors → $3^K$ combinations
  - Factor exposures: $e_k \in \{-1, 0, +1\}$ (short, neutral, long)
  - Target weights: $w_k = e_k \cdot \frac{\sigma_{\text{target}}}{\sigma_k}$ (risk parity scaling)
  - Portfolio return: $r_{p,t} = \sum_{k=1}^K w_k r_{k,t}$ where $r_{k,t}$ are factor returns
  - Turnover: $\text{TO}_t = \sum_k |w_{k,t} - w_{k,t-1}|$
  - Diversification: $N_{\text{active}} = \sum_k \mathbb{1}_{w_k \neq 0}$
  - Reward: $R_t = \Delta \text{PnL} + 0.1 \frac{N_{\text{active}}}{K} - 0.2 \mathbb{1}_{\max_k |w_k| > 0.5} - \frac{\text{TO}_t}{10000}$
- Strategy: discrete factor tilts (long/neutral/short) across Value/Momentum/Quality factors
- Market effect: factor momentum + correlation shifts change optimal tilt
- Diagram: drawio/factor_tracking.drawio

Slide 12 - Volatility Trading (PPO)
- Math: Black-Scholes Greeks with delta-hedged vega positions
  - $d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$, $d_2 = d_1 - \sigma\sqrt{T}$
  - Delta: $\Delta = N(d_1)$, Gamma: $\Gamma = \frac{N'(d_1)}{S\sigma\sqrt{T}}$
  - Vega: $\nu = S \cdot N'(d_1) \sqrt{T}$, Theta: $\Theta = -\frac{S \cdot N'(d_1) \sigma}{2\sqrt{T}} - rK e^{-rT}N(d_2)$
  - Reward: $R_t = \Delta PnL - \lambda_1 \cdot \text{HedgeCost} - \lambda_2 \cdot \mathbb{1}_{|\text{vega}|<10}$
- Strategy: delta-hedged vol positions, term-structure signals, IV-RV spread exploitation
- Market effect: IV-RV gap, vol-of-vol, and event risk drive actions
- Diagram: drawio/volatility_trading.drawio

Slide 13 - Delta Hedging (DDPG)
- Math: Optimal hedge ratio to minimize delta exposure and capture gamma
  - Theoretical delta: $\Delta_t = N(d_1) \cdot N_{\text{options}}$
  - Hedge shares: $h_t = \alpha \cdot \Delta_t$ where $\alpha \in [-1.5, 1.5]$
  - Hedge error: $\epsilon_t = h_t - \Delta_t$
  - Greeks: $\Gamma = \frac{N'(d_1)}{S\sigma\sqrt{T}}$, $\rho = KT e^{-rT} N(d_2)$
  - Reward: $R_t = \Delta PnL - \lambda_1 \frac{|\epsilon_t|}{10} - \lambda_2 \cdot \text{TransCost} + \lambda_3 \cdot \Gamma \cdot |\Delta S|$
- Strategy: manage Greeks (delta/gamma/vega) under transaction costs, gamma scalping
- Market effect: jumps and smile dynamics change hedge frequency
- Diagram: drawio/delta_hedging.drawio

Slide 14 - Futures Spreads (PPO)
- Math: Mean-reverting spread trading with z-score signals
  - Calendar spread: $S_t = F_{\text{far},t} - F_{\text{near},t}$
  - Inter-commodity: $S_t = F_{\text{far},t} / F_{\text{near},t}$
  - Z-score: $z_t = \frac{S_t - \mu_{S,w}}{\sigma_{S,w}}$ (window $w$)
  - Basis: $B_t = F_{\text{far},t} - F_{\text{near},t}$, Roll yield: $Y_t = B_t / F_{\text{near},t}$
  - Half-life: $\tau = -\ln(2) / \lambda$ from ADF test
  - Reward: $R_t = \Delta PnL + \lambda_1 \cdot \mathbb{1}_{|z_t|>2, \text{sign}(z_t \cdot \text{pos}_t)<0} - \lambda_2 \cdot \mathbb{1}_{|\text{pos}_t|<5}$
- Strategy: calendar/inter-commodity spreads, roll-yield signals, mean reversion
- Market effect: seasonality and carry change spread direction
- Diagram: drawio/futures_spreads.drawio

Slide 15 - FX Arbitrage (DDPG)
- Math: Triangular arbitrage and carry trade optimization
  - Triangular deviation: $\epsilon_{\triangle} = \frac{F_{\text{EUR/JPY}} - (F_{\text{EUR/USD}} \times F_{\text{USD/JPY}})}{F_{\text{EUR/USD}} \times F_{\text{USD/JPY}}}$
  - Carry: $C_i = r_{\text{quote},i} - r_{\text{base},i}$ (interest rate differential)
  - Position P&L: $\text{PnL}_i = \text{pos}_i \cdot (F_{i,t} - F_{i,t-1})$
  - Funding cost: $\text{FC}_t = \sum_i |\text{pos}_i| \cdot c_{\text{fund}} \cdot \Delta t$
  - Reward: $R_t = \Delta PnL + \lambda_1 \cdot |\epsilon_{\triangle}| \cdot 100 + \lambda_2 \sum_i \mathbb{1}_{\text{pos}_i \cdot C_i > 0} - \lambda_3 \cdot \text{FC}_t$
- Strategy: exploit triangular mispricings + carry; manage funding costs
- Market effect: rate differentials + liquidity shocks affect sizing
- Diagram: drawio/fx_arbitrage.drawio

Slide 16 - Agent Process (Per-Step Loop)
- Observe state -> normalize features -> policy outputs action
- Environment simulates fills, costs, constraints -> reward
- PPO on-policy updates; DDPG/DQN replay buffer off-policy updates

Slide 17 - Training Pipeline
- Phase 1: Train specialists independently (500K-1M steps each)
- Phase 2: Train CIO allocator with pretrained specialists (200K steps)
- Phase 3: Optional joint fine-tuning with lower specialist LR

Slide 18 - Decision Process Differences
- Continuous vs discrete actions define algorithm choice
- Strategy-specific penalties: inventory risk, hedge error, turnover
- Market regime features gate risk-taking and position size

Slide 19 - Results (Master CIO)
- Sharpe 1.60, Max Drawdown -0.80%, Annual Return 7.20%
- 6x Sharpe vs equal-weight; 78% lower drawdown
- Plots: reports/plots/equity_curves_comparison.png

Slide 20 - Specialist Results (Highlights)
- Stat Arb: 25.22% return, Sharpe 0.58
- Market Making: 17.66% return, Sharpe 0.85
- CIO allocation reduces exposure to underperformers

Slide 21 - Future Improvements
- Regime-aware reward shaping and adversarial stress tests
- Transaction cost calibration to venue-specific data
- Multi-task training for transfer across strategies
- Closing: hierarchical RL as a scalable fund architecture
