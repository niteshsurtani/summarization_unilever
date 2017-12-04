from flask import Flask, render_template,request
import os
import rake
import TextRank
import summary_LSA
import preprocessing
import text_rank_summary

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=tmpl_dir)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/keyphrase_extraction")
def keyphrase_extraction():
    return render_template("keyphrase.html")
@app.route("/keyphrase_extraction/textrank")
def keyphrase_extraction_textrank():
    return render_template("textrank.html")
@app.route("/summarization")
def text_summarization():
    return render_template("summarization.html")

@app.route("/summarization/textrank")
def text_summarization_textrank():
    return render_template("summarization_textrank.html")

@app.route("/get_summarization", methods = ['POST'])
def get_summarization():
    text = request.form['text']
    max_length_of_summary = request.form["max_length_of_summary"]
    number_of_concept = request.form["number_of_concept"]
    is_tfidf = request.form["is_tfidf"]
    split_long_sentence = request.form["split_long_sentence"]
    lsa_result = summary_LSA.summarize(text,l=max_length_of_summary,k=number_of_concept,tfidf=is_tfidf,to_split_length=split_long_sentence)
    context = dict()
    context['summarization'] = lsa_result
    return render_template("summarization_result.html", **context)

@app.route("/get_summarization_textrank", methods = ['POST'])
def get_summarization_textrank():
    text = request.form["text"]
    textRank_result = text_rank_summary.extract_sentences(text)
    context = dict()
    context["summarization"] = textRank_result
    return render_template("summarization_text_rank_result.html", **context)

@app.route('/get_keyphrases',methods=['POST'])
def get_keyphrases():
    stoppath = "SmartStoplist.txt"
    text=request.form['text']
    min_char_length=request.form['min_char_length']
    min_words_length=request.form['min_words_length']
    max_words_length=request.form['max_words_length']
    min_keyword_frequency=request.form['min_keyword_frequency']
    trade_off=request.form['trade_off']
    top_n=request.form['top_n']
    rake_object = rake.Rake(stoppath,int(min_char_length),int(min_words_length),int(max_words_length),int(min_keyword_frequency),1,3,2)
    keywords_score, keywords_counts, stem_counts = rake_object.run(text, float(trade_off),int(top_n))
    context = dict()
    context['keywords_score'] = keywords_score
    context['keywords_counts'] = keywords_counts
    context['stem_counts']= stem_counts
    return render_template("keyphrase_result.html", **context)

@app.route('/get_keyphrases_textrank',methods=['POST'])
def get_keyphrases_textrank():
    text=request.form['textrank_text']
    top_n=request.form['top_n_textrank']
    top_keywords=TextRank.extractKeyphrases(text,int(top_n))
    context=dict()
    context['keywords']=top_keywords
    return render_template("keyword_textrank.html",**context)

if __name__=="__main__":
    app.run(debug=True)
