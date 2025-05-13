# Simple script to save tournament results to a file
results_text = """
================================================================================
TOURNAMENT RESULTS
================================================================================

Board 1: [1, 3, 5, 7, 9]
  History first: History won 383/1000 (38.3%)
  No-history first: No-history won 480/1000 (48.0%)
  Combined: History won 903/2000 (45.1%)

Board 2: [2, 4, 4, 6, 9]
  History first: History won 508/1000 (50.8%)
  No-history first: No-history won 548/1000 (54.8%)
  Combined: History won 960/2000 (48.0%)

Board 3: [3, 3, 4, 6, 9]
  History first: History won 384/1000 (38.4%)
  No-history first: No-history won 622/1000 (62.2%)
  Combined: History won 762/2000 (38.1%)

Board 4: [1, 1, 3, 8, 12]
  History first: History won 767/1000 (76.7%)
  No-history first: No-history won 335/1000 (33.5%)
  Combined: History won 1432/2000 (71.6%)

Board 5: [5, 5, 5, 5, 5]
  History first: History won 449/1000 (44.9%)
  No-history first: No-history won 524/1000 (52.4%)
  Combined: History won 925/2000 (46.2%)

Board 6: [2, 2, 5, 7, 9]
  History first: History won 454/1000 (45.4%)
  No-history first: No-history won 569/1000 (56.9%)
  Combined: History won 885/2000 (44.2%)

Board 7: [1, 2, 3, 9, 10]
  History first: History won 153/1000 (15.3%)
  No-history first: No-history won 584/1000 (58.4%)
  Combined: History won 569/2000 (28.4%)

Board 8: [1, 1, 1, 8, 14]
  History first: History won 31/1000 (3.1%)
  No-history first: No-history won 877/1000 (87.7%)
  Combined: History won 154/2000 (7.7%)

Board 9: [3, 4, 6, 6, 6]
  History first: History won 431/1000 (43.1%)
  No-history first: No-history won 532/1000 (53.2%)
  Combined: History won 899/2000 (45.0%)

Board 10: [1, 1, 1, 1, 21]
  History first: History won 1000/1000 (100.0%)
  No-history first: No-history won 996/1000 (99.6%)
  Combined: History won 1004/2000 (50.2%)

Board 11: [1, 1, 1, 2, 20]
  History first: History won 20/1000 (2.0%)
  No-history first: No-history won 621/1000 (62.1%)
  Combined: History won 399/2000 (20.0%)

Board 12: [1, 2, 2, 5, 15]
  History first: History won 36/1000 (3.6%)
  No-history first: No-history won 782/1000 (78.2%)
  Combined: History won 254/2000 (12.7%)

--------------------------------------------------------------------------------
OVERALL RESULTS
History model wins: 9146/24000 (38.1%)
No-history model wins: 14854/24000 (61.9%)

PLAYER ORDER ANALYSIS
First player wins: 12086/24000 (50.4%)
History as first player: 4616/12000 (38.5%)
History as second player: 4530/12000 (37.8%)
"""

# Save to file
with open("tournament_results_history_vs_nohistory.txt", "w") as f:
    f.write(results_text)

print("Tournament results saved to: tournament_results_history_vs_nohistory.txt") 