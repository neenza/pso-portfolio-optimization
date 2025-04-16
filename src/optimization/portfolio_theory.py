import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Class for Modern Portfolio Theory calculations and traditional optimization methods.
    This will serve as a benchmark for comparison with PSO optimization.
    """
    
    def __init__(self, returns_data, risk_free_rate=0.0):
        """
        Initialize the optimizer with returns data
        
        Parameters:
        -----------
        returns_data : DataFrame
            DataFrame with asset returns (assets in columns, time in index)
        risk_free_rate : float, default=0.0
            Annual risk-free rate for Sharpe ratio calculation
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(returns_data.columns)
        self.assets = returns_data.columns
        self.risk_free_rate = risk_free_rate / 252  # Convert annual to daily
        
    def portfolio_return(self, weights):
        """
        Calculate portfolio return
        
        Parameters:
        -----------
        weights : array
            Array of weights for assets
            
        Returns:
        --------
        float
            Expected portfolio return
        """
        return np.sum(self.mean_returns * weights)
    
    def portfolio_volatility(self, weights):
        """
        Calculate portfolio volatility (standard deviation)
        
        Parameters:
        -----------
        weights : array
            Array of weights for assets
            
        Returns:
        --------
        float
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def sharpe_ratio(self, weights):
        """
        Calculate portfolio Sharpe ratio
        
        Parameters:
        -----------
        weights : array
            Array of weights for assets
            
        Returns:
        --------
        float
            Portfolio Sharpe ratio
        """
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol
    
    def neg_sharpe_ratio(self, weights):
        """
        Calculate negative Sharpe ratio (for minimization)
        
        Parameters:
        -----------
        weights : array
            Array of weights for assets
            
        Returns:
        --------
        float
            Negative Sharpe ratio
        """
        return -self.sharpe_ratio(weights)
    
    def min_variance(self, target_return=None):
        """
        Find minimum variance portfolio
        
        Parameters:
        -----------
        target_return : float, optional
            Target return constraint
            
        Returns:
        --------
        tuple
            (weights, expected_return, volatility, sharpe_ratio)
        """
        # Initial guess (equal weight)
        init_guess = np.ones(self.num_assets) / self.num_assets
        
        # Bounds and constraints
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Sum of weights = 1 constraint
        weights_sum_constraint = {'type': 'eq', 
                              'fun': lambda x: np.sum(x) - 1}
        
        constraints = [weights_sum_constraint]
        
        # Add target return constraint if specified
        if target_return is not None:
            return_constraint = {'type': 'eq',
                                'fun': lambda x: self.portfolio_return(x) - target_return}
            constraints.append(return_constraint)
        
        # Minimize volatility
        result = minimize(self.portfolio_volatility, init_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        optimal_weights = result['x']
        expected_return = self.portfolio_return(optimal_weights)
        volatility = self.portfolio_volatility(optimal_weights)
        sharpe = self.sharpe_ratio(optimal_weights)
        
        return optimal_weights, expected_return, volatility, sharpe
    
    def max_sharpe(self):
        """
        Find maximum Sharpe ratio portfolio
        
        Returns:
        --------
        tuple
            (weights, expected_return, volatility, sharpe_ratio)
        """
        # Initial guess (equal weight)
        init_guess = np.ones(self.num_assets) / self.num_assets
        
        # Bounds and constraints
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Sum of weights = 1 constraint
        weights_sum_constraint = {'type': 'eq', 
                              'fun': lambda x: np.sum(x) - 1}
        
        constraints = [weights_sum_constraint]
        
        # Maximize Sharpe ratio (minimize negative Sharpe)
        result = minimize(self.neg_sharpe_ratio, init_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        optimal_weights = result['x']
        expected_return = self.portfolio_return(optimal_weights)
        volatility = self.portfolio_volatility(optimal_weights)
        sharpe = self.sharpe_ratio(optimal_weights)
        
        return optimal_weights, expected_return, volatility, sharpe
    
    def efficient_frontier(self, points=50):
        """
        Calculate efficient frontier points
        
        Parameters:
        -----------
        points : int, default=50
            Number of points on the efficient frontier
            
        Returns:
        --------
        tuple
            (returns, volatilities, weights)
        """
        # Find minimum and maximum returns from individual assets
        min_ret = min(self.mean_returns)
        max_ret = max(self.mean_returns)
        
        # Create range of target returns
        target_returns = np.linspace(min_ret, max_ret, points)
        efficient_returns = []
        efficient_volatilities = []
        efficient_weights = []
        
        # Find minimum variance portfolio for each target return
        for target in target_returns:
            weights, ret, vol, _ = self.min_variance(target_return=target)
            
            if not np.isnan(vol):  # Check if optimization was successful
                efficient_returns.append(ret)
                efficient_volatilities.append(vol)
                efficient_weights.append(weights)
        
        return efficient_returns, efficient_volatilities, efficient_weights
    
    def plot_efficient_frontier(self, show_assets=True, show_min_vol=True, show_max_sharpe=True):
        """
        Plot the efficient frontier
        
        Parameters:
        -----------
        show_assets : bool, default=True
            Whether to show individual assets on the plot
        show_min_vol : bool, default=True
            Whether to highlight minimum volatility portfolio
        show_max_sharpe : bool, default=True
            Whether to highlight maximum Sharpe ratio portfolio
        """
        returns, volatilities, _ = self.efficient_frontier(points=50)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(volatilities, returns, 'b-', linewidth=3, label='Efficient Frontier')
        
        # Plot individual assets if requested
        if show_assets:
            asset_returns = self.mean_returns.values
            asset_volatilities = np.sqrt(np.diag(self.cov_matrix))
            plt.scatter(asset_volatilities, asset_returns, marker='o', s=100, 
                        color='red', label='Individual Assets')
            
            # Add asset labels
            for i, asset in enumerate(self.assets):
                plt.annotate(asset, (asset_volatilities[i], asset_returns[i]), 
                            xytext=(5, 5), textcoords='offset points')
        
        # Plot minimum volatility portfolio if requested
        if show_min_vol:
            min_vol_weights, min_vol_ret, min_vol_vol, _ = self.min_variance()
            plt.scatter(min_vol_vol, min_vol_ret, marker='*', color='green', s=200, 
                       label='Minimum Volatility')
        
        # Plot maximum Sharpe ratio portfolio if requested
        if show_max_sharpe:
            max_sharpe_weights, max_sharpe_ret, max_sharpe_vol, _ = self.max_sharpe()
            plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='orange', s=200, 
                       label='Maximum Sharpe Ratio')
        
        plt.title('Portfolio Optimization: Efficient Frontier')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def equal_weight_portfolio(self):
        """
        Calculate equal weight portfolio metrics
        
        Returns:
        --------
        tuple
            (weights, expected_return, volatility, sharpe_ratio)
        """
        weights = np.ones(self.num_assets) / self.num_assets
        expected_return = self.portfolio_return(weights)
        volatility = self.portfolio_volatility(weights)
        sharpe = self.sharpe_ratio(weights)
        
        return weights, expected_return, volatility, sharpe
    
    def portfolio_metrics_report(self):
        """
        Generate a report with metrics for different portfolio allocation strategies
        
        Returns:
        --------
        DataFrame
            Performance metrics for different strategies
        """
        # Get portfolio metrics for different strategies
        min_vol_weights, min_vol_ret, min_vol_vol, min_vol_sharpe = self.min_variance()
        max_sharpe_weights, max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe = self.max_sharpe()
        equal_weights, equal_ret, equal_vol, equal_sharpe = self.equal_weight_portfolio()
        
        # Create DataFrame for weights
        weights_df = pd.DataFrame({
            'Min Volatility': min_vol_weights,
            'Max Sharpe': max_sharpe_weights,
            'Equal Weight': equal_weights
        }, index=self.assets)
        
        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Min Volatility': [min_vol_ret, min_vol_vol, min_vol_sharpe],
            'Max Sharpe': [max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe],
            'Equal Weight': [equal_ret, equal_vol, equal_sharpe]
        }, index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        
        return weights_df, metrics_df

# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.data.data_processor import DataProcessor
    
    # Initialize data processor
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "5m")
    data_processor = DataProcessor(data_dir)
    
    # Load stock data (limit to 20 stocks for example)
    data_processor.load_stock_data(limit=20)
    
    # Calculate daily returns
    daily_returns = data_processor.calculate_daily_returns()
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(daily_returns, risk_free_rate=0.02)  # 2% risk-free rate
    
    # Calculate and print performance metrics
    weights_df, metrics_df = optimizer.portfolio_metrics_report()
    
    print("Portfolio Weights:")
    print(weights_df)
    print("\nPortfolio Metrics:")
    print(metrics_df)
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier()