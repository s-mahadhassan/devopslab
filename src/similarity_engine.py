import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_engine(df_model):
    """Builds and returns the scaled feature matrix and similarity querying function."""
    similarity_features = ['age', 'height', 'contract_duration_days']
    df_similarity = df_model.dropna(subset=similarity_features).copy()
    
    # Normalize features using Min-Max scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_similarity[similarity_features])
    
    # Compute pairwise cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(scaled_features, scaled_features)
    
    def find_similar_players(player_name, top_k=5):
        """Retrieves top K most similar players for a given player name."""
        if player_name not in df_similarity['player_name'].values:
            return f"Player '{player_name}' not found in dataset."
        
        # Get target index
        idx = df_similarity[df_similarity['player_name'] == player_name].index[0]
        pos_in_matrix = df_similarity.index.get_loc(idx)
        
        # Extract and sort similarity scores
        sim_scores = list(enumerate(cosine_sim_matrix[pos_in_matrix]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out the query player itself
        sim_scores = [score for score in sim_scores if score[0] != pos_in_matrix][:top_k]
        
        similar_indices = [i[0] for i in sim_scores]
        similar_scores = [i[1] for i in sim_scores]
        
        result_df = df_similarity.iloc[similar_indices][['player_name', 'main_position', 'age', 'height', 'current_club_name']].copy()
        result_df['similarity_score'] = similar_scores
        
        return result_df
        
    return find_similar_players, scaler