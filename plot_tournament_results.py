import argparse
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise tournament results")
    parser.add_argument("--input", type=str, default=None,
                        help="Tournament results file (default: most recent tournament_results_*.txt)")
    parser.add_argument("--output_prefix", type=str, default="tournament_viz",
                        help="Prefix for output image files")
    parser.add_argument("--plots", type=str, default="all",
                        help="Plot types to generate (comma-separated): bar,heatmap,first_mover,all")
    return parser.parse_args()

def find_results_file():
    """Find the most recent tournament results file"""
    files = [f for f in os.listdir('.') if f.startswith('tournament_results_') and f.endswith('.txt')]
    if not files:
        raise FileNotFoundError("No tournament results files found")
    
    # Sort by modification time (most recent first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def extract_tournament_data(filename):
    """Extract data from tournament results file"""
    boards = []
    history_first_wins = []
    history_second_wins = []
    nohistory_first_wins = []
    nohistory_second_wins = []
    history_win_rates = []
    
    with open(filename, 'r') as f:
        content = f.read()
        
        # Extract overall results
        overall_match = re.search(r'History model wins: (\d+)/(\d+) \(([0-9.]+)%\)', content)
        if overall_match:
            overall_history_wins = int(overall_match.group(1))
            overall_games = int(overall_match.group(2))
            overall_history_rate = float(overall_match.group(3))
        else:
            overall_history_wins, overall_games, overall_history_rate = 0, 0, 0
            
        # Extract first-player advantage
        first_player_match = re.search(r'First player wins: (\d+)/(\d+) \(([0-9.]+)%\)', content)
        if first_player_match:
            first_player_wins = int(first_player_match.group(1))
            total_games = int(first_player_match.group(2))
            first_player_rate = float(first_player_match.group(3))
        else:
            first_player_wins, total_games, first_player_rate = 0, 0, 0
            
        # Extract board-specific results
        board_pattern = r'Board \d+: \[([\d, ]+)\]\s+History first: History won (\d+)/(\d+) \(([0-9.]+)%\)\s+No-history first: No-history won (\d+)/(\d+) \(([0-9.]+)%\)\s+Combined: History won (\d+)/(\d+) \(([0-9.]+)%\)'
        
        for match in re.finditer(board_pattern, content):
            board = [int(x.strip()) for x in match.group(1).split(',')]
            boards.append(board)
            
            # History as first player
            history_first_total = int(match.group(3))
            history_first_rate = float(match.group(4))
            history_first_wins.append(int(match.group(2)))
            
            # No-history as first player
            nohistory_first_total = int(match.group(6))
            nohistory_first_rate = float(match.group(7))
            nohistory_first_wins.append(int(match.group(5)))
            
            # No-history as second player (against history first)
            nohistory_second_wins.append(history_first_total - int(match.group(2)))
            
            # History as second player (against no-history first)
            history_second_wins.append(nohistory_first_total - int(match.group(5)))
            
            # Overall history win rate for this board
            history_win_rates.append(float(match.group(10)))
    
    # Create a DataFrame for easier manipulation
    data = {
        'Board': [str(b) for b in boards],
        'Board_Index': list(range(len(boards))),
        'History_First_Wins': history_first_wins,
        'History_Second_Wins': history_second_wins,
        'NoHistory_First_Wins': nohistory_first_wins,
        'NoHistory_Second_Wins': nohistory_second_wins,
        'History_Win_Rate': history_win_rates,
        'NoHistory_Win_Rate': [100.0 - rate for rate in history_win_rates],
        'First_Player_Win_Rate': [(h_first + nh_first) / (h_first + nh_first + h_second + nh_second) * 100 
                                for h_first, nh_first, h_second, nh_second 
                                in zip(history_first_wins, nohistory_first_wins, 
                                      history_second_wins, nohistory_second_wins)]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate nimsum for each board
    df['Nimsum'] = [calculate_nimsum(eval(board)) for board in df['Board']]
    
    # Calculate the maximum pile size in each board
    df['Max_Pile'] = [max(eval(board)) for board in df['Board']]
    
    # Calculate the pile spread (max-min)
    df['Pile_Spread'] = [max(eval(board)) - min(eval(board)) for board in df['Board']]
    
    # Add board types based on known categories
    board_types = ['Training', 'Simple', 'Simple', 'Medium', 'Medium', 'Medium', 
                  'Hard', 'Hard', 'Hard', 'Extreme', 'Extreme', 'Extreme']
    if len(board_types) > len(df):
        board_types = board_types[:len(df)]
    elif len(board_types) < len(df):
        board_types.extend(['Unknown'] * (len(df) - len(board_types)))
    
    df['Board_Type'] = board_types
    
    # Add overall statistics
    overall_stats = {
        'overall_history_wins': overall_history_wins,
        'overall_games': overall_games,
        'overall_history_rate': overall_history_rate,
        'first_player_wins': first_player_wins,
        'total_games': total_games,
        'first_player_rate': first_player_rate
    }
    
    return df, overall_stats

def calculate_nimsum(board):
    """Calculate the nim-sum of a board"""
    result = 0
    for pile in board:
        result ^= pile
    return result

def plot_win_rate_bar(df, output_prefix):
    """Generate bar chart of history vs no-history win rates"""
    plt.figure(figsize=(14, 8))
    
    # Sort by history win rate
    df_sorted = df.sort_values('History_Win_Rate', ascending=False).reset_index(drop=True)
    
    bar_width = 0.35
    index = np.arange(len(df_sorted))
    
    # Create bars
    plt.bar(index, df_sorted['History_Win_Rate'], bar_width, label='History Enabled', color='blue', alpha=0.7)
    plt.bar(index + bar_width, df_sorted['NoHistory_Win_Rate'], bar_width, label='History Disabled', color='orange', alpha=0.7)
    
    # Add a horizontal line at 50%
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Even matchup (50%)')
    
    # Add board labels and formatting
    plt.xlabel('Board Configuration', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.title('Win Rates: History vs. No-History Models Across Board Configurations', fontsize=14)
    plt.xticks(index + bar_width/2, df_sorted['Board'], rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add board type annotations
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        plt.annotate(row['Board_Type'], 
                    xy=(i, 5), 
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='black', fontsize=8)
    
    plt.tight_layout()
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    filename = f"{output_prefix}_win_rate_bar.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Win rate bar chart saved to {filename}")
    plt.close()

def plot_heatmap(df, output_prefix):
    """Generate heatmap showing win rates by board and player role"""
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    heatmap_data = []
    for _, row in df.iterrows():
        board_str = row['Board']
        
        # Calculate win percentages
        history_first_pct = row['History_First_Wins'] / (row['History_First_Wins'] + row['NoHistory_Second_Wins']) * 100
        history_second_pct = row['History_Second_Wins'] / (row['History_Second_Wins'] + row['NoHistory_First_Wins']) * 100
        
        # Create rows for each board
        heatmap_data.append({
            'Board': board_str,
            'Board Type': row['Board_Type'],
            'Player Role': 'History as First Player',
            'Win Rate': history_first_pct
        })
        
        heatmap_data.append({
            'Board': board_str,
            'Board Type': row['Board_Type'],
            'Player Role': 'History as Second Player',
            'Win Rate': history_second_pct
        })
    
    #for seaborn
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Generate the heatmap using seaborn
    plt.figure(figsize=(14, 10))
    heatmap_pivot = heatmap_df.pivot(index='Board', columns='Player Role', values='Win Rate')
    
    # Sort by the average win rate
    heatmap_pivot['Average'] = heatmap_pivot.mean(axis=1)
    heatmap_pivot = heatmap_pivot.sort_values('Average', ascending=False)
    heatmap_pivot = heatmap_pivot.drop('Average', axis=1)
    
    # Create heatmap
    ax = sns.heatmap(heatmap_pivot, annot=True, cmap="RdYlGn", vmin=0, vmax=100, fmt=".1f",
                    linewidths=0.5, cbar_kws={'label': 'Win Rate (%)'})
    
    plt.title('History Model Win Rates by Board and Player Role', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    filename = f"{output_prefix}_win_rate_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Win rate heatmap saved to {filename}")
    plt.close()
    
    # Create another heatmap with board properties and performance
    plt.figure(figsize=(12, 10))
    
    # Prepare correlation data
    corr_data = df[['History_Win_Rate', 'First_Player_Win_Rate', 'Nimsum', 'Max_Pile', 'Pile_Spread']]
    
    # correlation matrix
    corr = corr_data.corr()
    
    # Create heatmap for correlations
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, 
               vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    
    plt.title('Correlation Between Board Properties and Performance', fontsize=14)
    plt.tight_layout()
    
    # Save the correlation heatmap
    filename = f"{output_prefix}_correlation_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to {filename}")
    plt.close()

def plot_first_mover_advantage(df, overall_stats, output_prefix):
    """Generate visualisation of first-mover advantage"""
    plt.figure(figsize=(14, 8))
    
    # Sort by first mover advantage
    df_sorted = df.sort_values('First_Player_Win_Rate', ascending=False).reset_index(drop=True)
    
    # Create bars
    bars = plt.bar(df_sorted['Board'], df_sorted['First_Player_Win_Rate'], 
                  color=plt.cm.viridis(df_sorted['First_Player_Win_Rate']/100))
    
    # Add a horizontal line at 50%
    plt.axhline(y=50, color='red', linestyle='--', label='No advantage (50%)')
    
    # Add overall first-mover rate
    plt.axhline(y=overall_stats['first_player_rate'], color='black', linestyle='-', 
               label=f'Overall first-mover rate ({overall_stats["first_player_rate"]:.1f}%)')
    
    # Add formatting
    plt.xlabel('Board Configuration', fontsize=12)
    plt.ylabel('First Player Win Rate (%)', fontsize=12)
    plt.title('First-Mover Advantage Across Board Configurations', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)  # Leave room for text
    
    # Add win rate text on top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{df_sorted.iloc[i]["First_Player_Win_Rate"]:.1f}%',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    # Add nimsum annotations below each bar
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        plt.annotate(f'Nimsum: {row["Nimsum"]}', 
                   xy=(i, -5), 
                   xytext=(0, 0),
                   textcoords="offset points",
                   ha='center', va='top',
                   color='blue', fontsize=8)
    
    plt.tight_layout()
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    #Save the plot
    filename = f"{output_prefix}_first_mover_advantage.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"First-mover advantage plot saved to {filename}")
    plt.close()
    
    # Create a scatter plot of board properties vs. first-mover advantage
    plt.figure(figsize=(14, 10))
    
    # Create a multi-plot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot Max Pile vs First Player Win Rate
    sns.scatterplot(x='Max_Pile', y='First_Player_Win_Rate', 
                   data=df, ax=axes[0, 0], s=100, alpha=0.7)
    axes[0, 0].set_title('Max Pile Size vs. First Player Advantage')
    axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].grid(alpha=0.3)
    
    # Plot Pile Spread vs First Player Win Rate
    sns.scatterplot(x='Pile_Spread', y='First_Player_Win_Rate', 
                   data=df, ax=axes[0, 1], s=100, alpha=0.7)
    axes[0, 1].set_title('Pile Size Spread vs. First Player Advantage')
    axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(alpha=0.3)
    
    # Plot Nimsum vs First Player Win Rate
    sns.scatterplot(x='Nimsum', y='First_Player_Win_Rate', 
                   data=df, ax=axes[1, 0], s=100, alpha=0.7)
    axes[1, 0].set_title('Nimsum vs. First Player Advantage')
    axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot History Win Rate vs First Player Win Rate
    sns.scatterplot(x='History_Win_Rate', y='First_Player_Win_Rate', 
                   data=df, ax=axes[1, 1], s=100, alpha=0.7)
    axes[1, 1].set_title('History Win Rate vs. First Player Advantage')
    axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save the scatter plots
    filename = f"{output_prefix}_first_mover_scatter.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"First-mover scatter plots saved to {filename}")
    plt.close()

def plot_board_type_comparison(df, output_prefix):
    """Generate comparison of performance by board type"""
    # Aggregate by board type
    board_type_stats = df.groupby('Board_Type').agg({
        'History_Win_Rate': 'mean',
        'NoHistory_Win_Rate': 'mean',
        'First_Player_Win_Rate': 'mean'
    }).reset_index()
    
    # Sort by difficulty (assuming the order is Training, Simple, Medium, Hard, Extreme)
    difficulty_order = {'Training': 0, 'Simple': 1, 'Medium': 2, 'Hard': 3, 'Extreme': 4, 'Unknown': 5}
    board_type_stats['Difficulty'] = board_type_stats['Board_Type'].map(difficulty_order)
    board_type_stats = board_type_stats.sort_values('Difficulty').reset_index(drop=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar positions
    bar_width = 0.25
    index = np.arange(len(board_type_stats))
    
    # Create grouped bars
    plt.bar(index - bar_width, board_type_stats['History_Win_Rate'], bar_width, 
           label='History Win Rate', color='blue', alpha=0.7)
    plt.bar(index, board_type_stats['NoHistory_Win_Rate'], bar_width, 
           label='No-History Win Rate', color='orange', alpha=0.7)
    plt.bar(index + bar_width, board_type_stats['First_Player_Win_Rate'], bar_width, 
           label='First Player Advantage', color='green', alpha=0.7)
    
    # Add horizontal line at 50%
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    
    # Add formatting
    plt.xlabel('Board Type', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.title('Performance Metrics by Board Type', fontsize=14)
    plt.xticks(index, board_type_stats['Board_Type'])
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    filename = f"{output_prefix}_board_type_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Board type comparison saved to {filename}")
    plt.close()

def main():
    args = parse_args()
    
    #find tournament results file
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return
        input_file = args.input
    else:
        try:
            input_file = find_results_file()
            print(f"Using most recent tournament results file: {input_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Extract data from the file
    try:
        df, overall_stats = extract_tournament_data(input_file)
    except Exception as e:
        print(f"Error parsing tournament results: {e}")
        return
    
    # Determine which plots to generate
    plot_types = args.plots.lower().split(',')
    generate_all = 'all' in plot_types
    
    # Generate requested plots
    if generate_all or 'bar' in plot_types:
        plot_win_rate_bar(df, args.output_prefix)
    
    if generate_all or 'heatmap' in plot_types:
        plot_heatmap(df, args.output_prefix)
    
    if generate_all or 'first_mover' in plot_types:
        plot_first_mover_advantage(df, overall_stats, args.output_prefix)
    
    if generate_all or 'board_type' in plot_types:
        plot_board_type_comparison(df, args.output_prefix)
    
    #Print summary
    print(f"\nSummary of tournament results:")
    print(f"- Overall win rate for history model: {overall_stats['overall_history_rate']:.1f}%")
    print(f"- First-player advantage across all games: {overall_stats['first_player_rate']:.1f}%")
    print(f"- Total games played: {overall_stats['overall_games']}")
    
    # Calculate performance by board type
    board_type_stats = df.groupby('Board_Type')['History_Win_Rate'].mean()
    print("\nHistory model performance by board type:")
    for board_type, win_rate in board_type_stats.items():
        print(f"- {board_type}: {win_rate:.1f}%")

if __name__ == "__main__":
    main() 