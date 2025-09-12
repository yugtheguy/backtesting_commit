# api/views.py

import json
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .engine import UniversalStrategyParser, HybridBacktester, HybridVisualizationEngine

# Define the path to your CSV file
DATA_FILE_PATH = 'aapl_data.csv'

class BacktestAPIView(APIView):
    """
    API endpoint to run a backtest.
    Accepts a POST request with a 'strategy' JSON object.
    """
    def post(self, request, *args, **kwargs):
        try:
            # 1. Get Strategy JSON from request body
            strategy_json = request.data
            if not isinstance(strategy_json, dict):
                strategy_json = json.loads(strategy_json.get('strategy'))

            print(f"Strategy received: {strategy_json.get('name', 'Unnamed')}")

            # 2. Load the historical data from the local file
            try:
                data = pd.read_csv(DATA_FILE_PATH)
                print(f"Data loaded: {len(data)} rows, from {data['Date'].min()} to {data['Date'].max()}")
                
                # Validate required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    return Response(
                        {"error": f"Missing required columns: {missing_columns}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                    
            except FileNotFoundError:
                return Response(
                    {"error": f"Data file not found at {DATA_FILE_PATH}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # 3. Parse the strategy
            try:
                parser = UniversalStrategyParser(strategy_json)
                parsed_strategy = parser.parse()
                print(f"Strategy parsed: {len(parsed_strategy['strategy_groups'])} groups, {len(parsed_strategy['indicators'])} indicators")
            except Exception as e:
                return Response(
                    {"error": f"Error parsing strategy: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 4. Run the backtest
            try:
                backtester = HybridBacktester(initial_capital=100000)
                results = backtester.run_backtest(data, parsed_strategy)
                
                print(f"Backtest completed:")
                print(f"  - Final capital: ${results['metrics']['final_capital']:,.2f}")
                print(f"  - Total trades: {results['metrics']['total_trades']}")
                print(f"  - Equity curve points: {len(results['equity_curve'])}")
                
            except Exception as e:
                print(f"Backtesting error: {str(e)}")
                return Response(
                    {"error": f"Error during backtesting: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # 5. Generate the interactive plot HTML
            try:
                plot_html = HybridVisualizationEngine.create_single_interactive_plot(results, data)
                print(plot_html)
                print("Plot generated successfully")
            except Exception as e:
                print(f"Plot generation error: {str(e)}")
                plot_html = f"<div>Plot generation failed: {str(e)}</div>"
            
            # 6. Construct and send the final response
            final_response = {
            'metrics': results.get('metrics', {}),
            'plot_html': plot_html,
            'suggestions': results.get('suggestions', []), # Pass the suggestions from the results
            
            }
            return Response(final_response, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return Response({"error": "Invalid JSON format in strategy."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return Response({"error": f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)