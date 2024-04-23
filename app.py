import flask
import pandas as pd
import joblib
import math


G = joblib.load('g.pkl')
app = flask.Flask(__name__, template_folder='templates')


df2 = pd.read_csv('dt_df.csv')


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
    
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html', all_movies=all_titles))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name'].strip()

        if m_name not in all_titles:
            return(flask.render_template('notFound.html', name=m_name, all_movies = all_titles))
        else:
            result_final = recommend(m_name)
            names = result_final['title'].tolist()
            return flask.render_template('found.html', movie_names=names, search_name = m_name)

if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=8080, debug=False)
    app.run(debug=True)
