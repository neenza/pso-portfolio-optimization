# Portfolio Optimization

A Python-based portfolio optimization tool that implements Particle Swarm Optimization (PSO) approach.

## Installation
```bash
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization
pip install -r requirements.txt

## Usage
python main.py --data_dir "5m" --n_stocks 30 --risk_free_rate 0.02 --objective sharpe

## Arguments 
- --data_dir: Directory containing stock data CSV files
- --n_stocks: Number of stocks to include in portfolio (default: 30)
- --risk_free_rate: Annual risk-free rate (default: 0.02 or 2%)
- --objective: Optimization objective (choices: sharpe, min_risk, max_return, calmar, sortino)
- --n_particles: Number of particles in PSO (default: 100)
- --iters: Number of iterations in PSO (default: 200)
- --output_dir: Directory to save results (default: results)
