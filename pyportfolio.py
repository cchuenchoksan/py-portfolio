import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from scipy.optimize import minimize
import requests
from bs4 import BeautifulSoup


class PyPortfolio:
    def __init__(self, tickers: list, rf=None, start="2014-01-01", end="2023-12-31"):
        if rf is None:
            self.rf = self.get_sg_risk_free_rate()
        else:
            self.rf = rf
        
        self.update_tickers(tickers, start, end)
        self._get_returns_and_cov()

        # Private for optimisation
        self._eq_constraint = {'type': 'eq', 'fun': self._equality_constraint}
        self._initial_weights = np.array([1/len(tickers) for _ in range(len(tickers))])
        self._bounds = [(-10, 10) for _ in range(len(self._initial_weights))]

    def update_tickers(self, tickers, start, end):
        self.tickers = tickers
        self.start = start
        self.end = end

        self.info = self._get_info(tickers)
        self.data = yf.download(tickers, start=start, end=end)["Adj Close"]
        ten_years_return = ((self.data.iloc[-1] - self.data.iloc[0]) / self.data.iloc[0]) * 100
        self.info['10-Y Return(%)'] = self.info['Ticker'].map(ten_years_return)

    
    def _get_returns_and_cov(self):
        self.daily_r = self.data.pct_change().dropna()
        self.expected_returns = self.daily_r.mean() * 252
        self.cov_matrix = self.daily_r.cov() * 252


    def _objective_min_vol(self):
        return lambda weights: np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * 100


    def _objective_max_sharpe(self):
        return lambda weights: -((np.dot(weights, self.expected_returns) - self.rf) 
                                / np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))))

    
    @staticmethod
    def _equality_constraint(weights):
        return np.sum(weights) - 1


    @staticmethod
    def _generate_numbers(n: int):
        if n <= 1:
            raise ValueError("n must be greater than 1.")

        numbers = np.random.uniform(-1, 1, n - 1)
        nth_number = 1 - np.sum(numbers)
        result = np.empty(n)
        result[:-1] = numbers
        result[-1] = nth_number

        np.random.shuffle(result)
        return result


    def get_optimise_port(self, initial_weights=None):
        if initial_weights is None:
            initial_weights = self._initial_weights

        solution = minimize(self._objective_max_sharpe(), initial_weights,
                            constraints=self._eq_constraint, 
                            bounds=self._bounds, 
                            options={'disp': False})
        
        optimal_weights = solution.x
        optimal_returns = np.dot(optimal_weights, self.expected_returns)
        optimal_std = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        return optimal_weights, optimal_returns, optimal_std


    def get_lowest_vol_port(self, initial_weights=None):
        if initial_weights is None:
            initial_weights = self._initial_weights

        solution = minimize(self._objective_min_vol(), initial_weights,
                            constraints=self._eq_constraint, 
                            bounds=self._bounds, 
                            options={'disp': False})
        
        min_vol_weights = solution.x
        min_vol_returns = np.dot(min_vol_weights, self.expected_returns)
        min_vol_std = np.sqrt(np.dot(min_vol_weights.T, np.dot(self.cov_matrix, min_vol_weights)))
        return min_vol_weights, min_vol_returns, min_vol_std



    def run_monte_carlo_simulation(self, num_portfolios=10000):
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in tqdm(range(num_portfolios)):
            weights = self._generate_numbers(n=len(self.tickers))
            weights_record.append(weights)

            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.rf) / portfolio_stddev

            results[0,i] = portfolio_return
            results[1,i] = portfolio_stddev
            results[2,i] = sharpe_ratio

        return results, weights_record


    @staticmethod
    def get_sg_risk_free_rate():
        """Function to get the risk free rate from the singapore government website"""
        url = 'https://eservices.mas.gov.sg/statistics/fdanet/SgsBenchmarkIssuePrices.aspx'
        response = requests.get(url)
        
        if response.status_code != 200:
            print("Failed to set rf", response.status_code)
            print("Setting rf to 0.03")
            return 0.03

        soup = BeautifulSoup(response.content)
        table_html = str(soup.find_all("table")[0])

        df_rf = pd.read_html(table_html)[0]
        df_rf.head(10)
        return np.mean(df_rf["Treasury Bills"].iloc[-1].to_numpy()) / 100


    @staticmethod
    def _get_info(tickers: list):
        """Function that returns the stock information from the tickers"""
        info_list = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', 'N/A')

            info_list.append({
                'Ticker': ticker,
                'Company': info.get('longName', info.get('name', 'Unknown')),
                'Sector': info.get('sector', 'N/A'),
                'Market Cap (B SGD)': market_cap / 1e9 if market_cap != 'N/A' else 'N/A'
            })

        data = pd.DataFrame(info_list)
        return data

