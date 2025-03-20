import pandas as pd

# Load the top contenders file
try:
    top_teams = pd.read_csv("top_contenders_2025.csv")
    print("✅ Loaded top contenders from 'top_contenders_2025.csv'")
except FileNotFoundError:
    print("❌ Error: 'top_contenders_2025.csv' not found! Run 'train_and_predict_seeds.py' first.")
    exit()

# Sort teams by overall power ranking (BARTHAG)
top_teams = top_teams.sort_values(by="BARTHAG", ascending=False)

# Function to simulate matchups
def predict_matchup(team1, team2):
    """Predict winner based on efficiency metrics."""
    offense_advantage = team1["ADJOE"] - team2["ADJDE"]
    defense_advantage = team1["ADJDE"] - team2["ADJOE"]

    if offense_advantage > defense_advantage:
        return team1["TEAM"]
    else:
        return team2["TEAM"]

# Simulate the tournament (single-elimination)
round_1_winners = [predict_matchup(top_teams.iloc[i], top_teams.iloc[i+1]) for i in range(0, len(top_teams), 2)]
round_2_winners = [predict_matchup(top_teams.iloc[i], top_teams.iloc[i+1]) for i in range(0, len(round_1_winners), 2)]
champion = predict_matchup(top_teams.iloc[0], top_teams.iloc[1])

print(f"🏆 Predicted 2025 NCAA Champion: {champion}")
