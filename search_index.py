import pickle
from porter2stemmer import Porter2Stemmer
import tkinter as tk
import webbrowser
import math
import time


def main():
    window()


def window():
    term_data = pickle.load(open("term_data.pkl", 'rb'))
    names = pickle.load(open("doc_id_map.pkl", "rb"))
    index = open("final_index.txt", 'r')
    root = tk.Tk()
    canvas = tk.Canvas(root, height=700, width=800)
    canvas.pack()

    frame = tk.Frame(root, bg='white')
    frame.place(relwidth=1, relheight=1)
    entry = tk.Entry(frame)
    entry.place(relx=0.2,rely=0.3,relwidth=0.6,relheight=0.05)
    mylist = tk.Listbox(frame)
    mylist.place(relx=0.1,rely=0.5,relwidth=0.8,relheight=0.4)
    button = tk.Button(frame, text='Search', command=lambda: get_query(entry.get(), term_data, names, mylist, index))
    button.place(relx=0.4,rely=0.35,relwidth=0.2,relheight=0.06)

    root.bind("<Return>", (lambda event: get_query(entry.get(), term_data, names, mylist, index)))
    mylist.bind("<Double-Button-1>", lambda event: internet(mylist,event))

    background_image = tk.PhotoImage(file='google.png')
    background_label = tk.Label(frame, image=background_image)
    background_label.place(relx=0.26,rely=0.05,relwidth=0.48, relheight=0.18)

    root.mainloop()


def internet(listbox, event):
    link = listbox.get(listbox.nearest(event.y))
    webbrowser.open(link)


def get_query(query, term_positions, doc_ids, list_box, index):
    t0 = time.time()
    list_box.delete(0,'end')
    results = search_index(query, term_positions, index)
    t1 = time.time()

    if results == []:
        print("No results for query:", query)
        return

    print("Time:", (t1 - t0) * 1000, "milliseconds.")

    for i in range(len(results)):
        if i > 9:
            break
        list_box.insert('end', doc_ids[results[i]])


def search_index(query, term_data, index_file):
    tokens = [porter_stemmer(word) for word in query.lower().strip().split(" ")]
    token_to_tfidf = dict()
    document_to_token = dict(dict())
    token_postings = dict()
    sorted_tokens = []
    if len(tokens) > 1:
        for token in tokens:
            try:
                position = term_data[token][0]
                idf = term_data[token][1]
                if idf > 0.7:
                    index_file.seek(position)
                    posting = eval(index_file.readline().split(":")[1])
                    tf = 1 + math.log(tokens.count(token), 10)
                    tf_idf = tf * idf
                    token_to_tfidf[token] = tf_idf
                    token_postings[token] = posting
                    sorted_tokens.append(token)
            except KeyError:
                print("No match for term:", token)

        for term in sorted_tokens:
            for document in token_postings[term]:
                if document[0] in document_to_token:
                    document_to_token[document[0]][term] = document[3]
                else:
                    document_to_token[document[0]] = {term: document[3]}
        cosine_similarities = []
        for document in document_to_token:
            sim = cos_sim(token_to_tfidf, document_to_token[document])
            cosine_similarities.append((document, sim))

    else:
        token = tokens[0]
        try:
            position = term_data[token][0]
            index_file.seek(position)
            posting = eval(index_file.readline().split(":")[1])
            return [x[0] for x in sorted(posting, key=lambda y: y[3], reverse=True)]
        except KeyError:
            print("No match for term:", token)
            return []

    return [x[0] for x in sorted(cosine_similarities, key=lambda y:y[1], reverse=True)]


def porter_stemmer(word):
    stemmer = Porter2Stemmer()
    return stemmer.stem(word)


def cos_sim(query_dict, document_dict):
    total = 0
    query_norm = normalize(query_dict)
    document_dict = normalize(document_dict)
    for term in query_norm:
        if term in document_dict:
            total += query_norm[term] * document_dict[term]
    return total


def normalize(tf_idf_dict):
    total = 0
    for term in tf_idf_dict:
        total += tf_idf_dict[term]**2
    total = math.sqrt(total)

    for term in tf_idf_dict:
        tf_idf_dict[term] = tf_idf_dict[term]/total

    return tf_idf_dict

main()



