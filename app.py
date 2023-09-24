from flask import Flask, render_template, request, jsonify
import clustering

app = Flask(__name__)

# Create a route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Create a route to handle the form submission
@app.route('/find_similar_anime', methods=['POST'])
def find_similar_anime_route():
    anime_name = request.form.get('animeName')
    anime_name = anime_name.strip().lower()
    similar_anime = clustering.find_similar_anime(anime_name)
    
    # Pass the similar_anime data to the template
    return render_template('index.html', similar_anime=similar_anime)

if __name__ == '__main__':
    app.run(debug=True)
