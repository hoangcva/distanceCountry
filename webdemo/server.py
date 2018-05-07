from flask import Flask, request, render_template, Markup
from visualize import visualize
from similarity_words import list_country, distance_country_euclidean

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # my_plot_div = visualize()
    return render_template('index.html',
                           div_placeholder=Markup(visualize()), countries=list_country()
                           )

@app.route('/method/<namemethod>', methods=['GET', 'POST'])
def home(namemethod):
    # my_plot_div = visualize(namemethod)
    print namemethod
    return render_template('index.html',
                            div_placeholder=Markup(visualize(namemethod)), countries=list_country(namemethod)
                           )

@app.route('/query', methods=['POST'])
def result():
    data = request.get_json(force=True)
    result2 = distance_country_euclidean(data["country1"], data["country2"], namemethod=data["methodw2v"])
    return result2


if __name__ == '__main__':
    app.run(debug=True)
