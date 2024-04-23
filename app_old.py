import flask
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import joblib
import math

# dt= joblib.load('decisionTree.pkl')
G = joblib.load('g.pkl')
app = flask.Flask(__name__, template_folder='templates')

# df2 = pd.read_csv(r'model\tmdb_5000_movies.csv')
df2 = pd.read_csv('dt_df.csv')

# tfidf = TfidfVectorizer(stop_words='english', analyzer='word')

# # Construct the required TF-IDF matrix by fitting and transforming the data
# tfidf_matrix = tfidf.fit_transform(df2['soup'])
# print(tfidf_matrix.shape)

# # Construct cosine similarity matrix
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print(cosine_sim.shape)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Create array with all movie titles
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

def recommend(root):
    commons_neighbours = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2 == root:
                continue
            if G.nodes[e2]['label'] == "MOVIE":
                commons = commons_neighbours.get(e2)
                if commons is None:
                    commons_neighbours[e2] = [e]
                else:
                    commons.append(e)
                    commons_neighbours[e2] = commons

    movies = []
    weight = []

    for key, values in commons_neighbours.items():
        w = 0.0
        for e in values:
            w = w + 1 / math.log(G.degree(e))
        movies.append(key)
        weight.append(w)

    result = pd.DataFrame(data={'title': movies, 'weight': weight})
    result.sort_values(by='weight', inplace=True, ascending=False)
    return result
    
# def get_recommendations(title):
#     # Get the index of the movie that matches the title
#     idx = indices[title]
#     # Get the pairwise similarity scores of all movies with that movie
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     # Sort the movies based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     # Get the scores of the 10 most similar movies
#     sim_scores = sim_scores[1:11]

#     # Get the movie indices
#     movie_indices = [i[0] for i in sim_scores]

#     # Return list of similar movies
#     return_df = pd.DataFrame(columns=['Title', 'Homepage'])
#     return_df['Title'] = df2['title'].iloc[movie_indices]
#     return_df['Homepage'] = df2['homepage'].iloc[movie_indices]
#     return_df['ReleaseDate'] = df2['release_date'].iloc[movie_indices]
#     return return_df

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html', all_movies=all_titles))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name'].strip()  # Remove leading/trailing whitespaces
        # Check if the movie name exists in the dataset
        if m_name not in all_titles:
            return(flask.render_template('notFound.html', name=m_name, all_movies = all_titles))
        else:
            result_final = recommend(m_name)
            names = result_final['title'].tolist()
            # homepage = result_final['Homepage'].tolist()
            # releaseDate = result_final['ReleaseDate'].tolist()
            # return flask.render_template('found.html', movie_names=names, movie_homepage=homepage,
            #                              search_name=m_name, movie_releaseDate=releaseDate)
            return flask.render_template('found.html', movie_names=names, search_name = m_name)

if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=8080, debug=False)
    app.run(debug=True)
