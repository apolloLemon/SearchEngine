# -*- coding:utf-8 -*-
import os
import sys
import searcher
from flask import Flask, render_template, request, jsonify
 



# create flask instance
app = Flask(__name__)




# main route
@app.route('/')
def index():
    return render_template('./index.html')

# search route
@app.route('/search', methods=['POST'])
def search():
    
    if request.method == "POST":
        query = request.form.get('query')
        result, texts = searcher.searchDoc(query)
        return render_template("result.html", len = len(result), results = result, texts = texts, query = query) 


if __name__ == '__main__':
    app.run('127.0.0.1', debug=True)