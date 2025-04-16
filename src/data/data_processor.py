import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    """
    Class to handle data loading, cleaning, and preprocessing for portfolio optimization
    """
    
    def __init__(self, data_dir):
        """
        Initialize DataProcessor with the directory containing stock data
        
        Parameters:
        -----------
        data_dir : str
            Path to directory containing CSV files with stock data
        """
        self.data_dir = data_dir
        self.stock_data = {}
        self.returns_data = None
        self.daily_returns = None
        
    def load_stock_data(self, limit=None):
        """
        Load all stock data from CSV files in the data directory
        
        Parameters:
        -----------
        limit : int, optional
            Limit the number of stocks to load for testing purposes
        
        Returns:
        --------
        dict
            Dictionary with stock symbols as keys and DataFrames as values
        """
        print(f"Loading stock data from {self.data_dir}")
        
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if limit:
            all_files = all_files[:limit]
            
        for file in all_files:
            stock_symbol = file.split('.')[0]
            file_path = os.path.join(self.data_dir, file)
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'])
                
                # Set time as index
                df.set_index('time', inplace=True)
                
                # Store data in dictionary with stock symbol as key
                self.stock_data[stock_symbol] = df
                
                print(f"Loaded {stock_symbol} data: {len(df)} rows")
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        return self.stock_data
    
    def calculate_returns(self, price_col='close'):
        """
        Calculate returns for all stocks based on specified price column
        
        Parameters:
        -----------
        price_col : str, default='close'
            Which price column to use for return calculations
            
        Returns:
        --------
        DataFrame
            DataFrame with returns for all stocks
        """
        if not self.stock_data:
            raise ValueError("Stock data has not been loaded. Call load_stock_data() first.")
        
        # Create dictionary to store returns series
        returns_dict = {}
        
        for symbol, df in self.stock_data.items():
            # Calculate percentage returns
            returns_dict[symbol] = df[price_col].pct_change().fillna(0)
        
        # Combine all returns into a single DataFrame
        self.returns_data = pd.DataFrame(returns_dict)
        
        return self.returns_data
    
    def calculate_daily_returns(self):
        """
        Resample 5-minute returns to daily returns
        
        Returns:
        --------
        DataFrame
            DataFrame with daily returns for all stocks
        """
        if self.returns_data is None:
            self.calculate_returns()
        
        # Resample to daily returns (business days)
        self.daily_returns = self.returns_data.resample('B').sum()
        
        return self.daily_returns
    
    def plot_returns_distribution(self, n_stocks=5):
        """
        Plot the distribution of returns for a sample of stocks
        
        Parameters:
        -----------
        n_stocks : int, default=5
            Number of stocks to include in the plot
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        # Sample n stocks
        sample_stocks = np.random.choice(self.daily_returns.columns, min(n_stocks, len(self.daily_returns.columns)), replace=False)
        
        # Plot distribution of returns
        plt.figure(figsize=(12, 8))
        for stock in sample_stocks:
            sns.histplot(self.daily_returns[stock].dropna(), kde=True, label=stock)
        
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    def calculate_correlation_matrix(self):
        """
        Calculate correlation matrix for daily returns
        
        Returns:
        --------
        DataFrame
            Correlation matrix for all stocks
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        return self.daily_returns.corr()
    
    def plot_correlation_heatmap(self, n_stocks=20):
        """
        Plot correlation heatmap for a sample of stocks
        
        Parameters:
        -----------
        n_stocks : int, default=20
            Number of stocks to include in the heatmap
        """
        corr_matrix = self.calculate_correlation_matrix()
        
        # Sample n stocks
        sample_stocks = np.random.choice(corr_matrix.columns, min(n_stocks, len(corr_matrix.columns)), replace=False)
        sample_corr = corr_matrix.loc[sample_stocks, sample_stocks]
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(sample_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Daily Returns')
        plt.tight_layout()
        plt.show()
    
    def calculate_statistics(self):
        """
        Calculate key statistics for each stock
        
        Returns:
        --------
        DataFrame
            DataFrame with statistics for all stocks
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        # Calculate statistics
        mean_returns = self.daily_returns.mean()
        std_returns = self.daily_returns.std()
        sharpe = mean_returns / std_returns  # Simplified Sharpe ratio (no risk-free rate)
        
        # Create DataFrame with statistics
        stats_df = pd.DataFrame({
            'Mean Return': mean_returns,
            'Volatility': std_returns,
            'Sharpe Ratio': sharpe
        })
        
        return stats_df
    
    def split_train_test(self, test_ratio=0.2):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        test_ratio : float, default=0.2
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (train_data, test_data)
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        # Calculate split point
        n = len(self.daily_returns)
        split_idx = int(n * (1 - test_ratio))
        
        # Split data
        train_data = self.daily_returns.iloc[:split_idx]
        test_data = self.daily_returns.iloc[split_idx:]
        
        return train_data, test_data

# Example usage
if __name__ == "__main__":
    # Initialize data processor with path to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "5m")
    processor = DataProcessor(data_dir)
    
    # Load stock data (limit to 10 stocks for testing)
    processor.load_stock_data(limit=10)
    
    # Calculate returns
    processor.calculate_returns()
    
    # Calculate daily returns
    processor.calculate_daily_returns()
    
    # Calculate and print statistics
    stats = processor.calculate_statistics()
    print("\nStock Statistics:")
    print(stats)
    
    # Plot returns distribution
    processor.plot_returns_distribution()
    
    # Plot correlation heatmap
    processor.plot_correlation_heatmap()
    
    # Split data into training and testing sets
    train_data, test_data = processor.split_train_test()
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")