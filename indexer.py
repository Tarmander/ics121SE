import os
import json
import re
import pickle
import math
from bs4 import BeautifulSoup
from porter2stemmer import Porter2Stemmer
from simhash import Simhash
import linecache
MAX_DOCS = 0


def main():
    prompt = input("Are you sure you want to restart the index?")
    if prompt.lower() == "yes":
        reset_index()
        num_indexes = index_creator()
        merge_index(num_indexes)
    else:
        return


def merge_index(index_size):
    for index in range(1, index_size + 1):
        print("Merging index", str(index))
        temp_index = open("temp_merge.txt", "w")
        str_merged_index = "index0.txt"
        str_next_index = "index" + str(index) + ".txt"
        i = j = 1
        while (True):
            print(i, j)
            merged = linecache.getline(str_merged_index, i)
            next = linecache.getline(str_next_index, j)
            if merged != '':
                merged_term, merged_posting_list = linecache.getline(str_merged_index, i).split(":")
                merged_posting_list = eval(merged_posting_list)
            else:
                merged_term = ''

            if next != '':
                next_term, next_posting_list = linecache.getline(str_next_index, j).split(":")
                next_posting_list = eval(next_posting_list)
            else:
                next_term = ''

            if merged_term == '' and next_term == '':
                break
            if merged_term == next_term and merged_term != '':
                merged_posting_list.extend(next_posting_list)
                temp_index.write("{}:{}\n".format(merged_term, merged_posting_list))
                i += 1
                j += 1
            elif merged_term != '' and (merged_term < next_term or next_term == ''):
                temp_index.write("{}:{}\n".format(merged_term, merged_posting_list))
                i += 1
            elif next_term != '' and (next_term < merged_term or merged_term == ''):
                temp_index.write("{}:{}\n".format(next_term, next_posting_list))
                j += 1
        temp_index.close()
        merged_index = open(str_merged_index, "w")
        temp_index = open("temp_merge.txt",'r')
        for line in temp_index:
            merged_index.write(line)
        merged_index.close()
        temp_index.close()
    print("constructing final index")
    merged_index = open("index0.txt","r")
    final_index = open("final_index.txt", "w+")
    term_data = dict()
    for line in merged_index:
        term, posting_list = line.split(":")
        posting_list = eval(posting_list)
        posting_list = get_tfidf(posting_list)
        term_data[term] = (final_index.tell(), math.log(MAX_DOCS/len(posting_list), 10))
        final_index.write("{}:{}\n".format(term, posting_list))
    pickle.dump(term_data, open("term_data.pkl", "wb"))
    print("finished")


def index_creator():
    current_doc = 1
    doc_dict = dict()
    word_dict = dict()
    hashes = set()
    unique_urls = set()
    count = 0
    for root, dirs, files in os.walk("DEV"):
        for name in files:
            with open(os.path.join(root, name)) as json_file:
                data = json.load(json_file)
                words, title_terms, header_terms = parse_json_data(data, hashes)
                if words:
                    url = data['url']
                    defragged_url = url.split("#")[0]
                    if defragged_url not in unique_urls:
                        unique_urls.add(defragged_url)
                        doc_dict[current_doc] = url
                        unique_words = set(words)
                        doc_word_counts = create_word_count(words)
                        word_dict = create_word_dict(word_dict, unique_words, doc_word_counts, title_terms, header_terms,
                                                 current_doc)

                        if current_doc % 10000 == 0:
                            write_index(word_dict, count)
                            word_dict.clear()
                            count += 1
                        print(current_doc)
                        current_doc += 1
    write_index(word_dict, count)
    write_doc_id(doc_dict)
    global MAX_DOCS
    MAX_DOCS = current_doc
    return count


def porter_stemmer(word):
    stemmer = Porter2Stemmer()
    return stemmer.stem(word)


def get_tfidf(posting_list):
    global MAX_DOCS
    df = len(posting_list)
    idf = math.log(MAX_DOCS / df, 10)
    for i in range(len(posting_list)):
        posting = posting_list[i]
        tf = posting[2]
        tfidf = tf * idf
        posting_list[i][3] = tfidf
    return posting_list


def parse_json_data(json_data, hashlist):
    title_words = set()
    header_words = set()
    html = json_data['content']
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    words = [porter_stemmer(word) for word in re.split('[^a-z0-9]', text.lower()) if word != '']
    if not compare_hash(words, hashlist):
        return [], [], []

    for script in soup(["script", "style"]):
        script.extract()

    for title in soup.find_all(["title"]):
        text = title.get_text()
        for word in re.split('[^a-z0-9]', text.lower()):
            if word != '':
                title_words.add(porter_stemmer(word))

    for header in soup.find_all("h1","h2","h3","b","strong"):
        text = header.get_text()
        for word in re.split('[^a-z0-9]', text.lower()):
            if word != '':
                header_words.add(porter_stemmer(word))

    return words, title_words, header_words


def create_word_count(words):
    result = dict()
    for word in words:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1

    return result


def create_word_dict(current_dict, unique, word_counts, title_words, header_words, doc):
    for word in unique:
        tf = 1 + math.log(word_counts[word], 10)
        posting = [doc, word_counts[word], tf, None]
        if word in title_words:
            posting[2] *= 1.15
        elif word in header_words:
            posting[2] *= 1.05

        if word in current_dict:
            current_dict[word].append(posting)
        else:
            current_dict[word] = [posting]
    return current_dict


def compare_hash(content, hashlist):
    newhash = Simhash(content)
    for hash in hashlist:
        if newhash.distance(hash) <= 2:
            return False
    hashlist.add(newhash)
    return True


def write_index(words, index_num):
    index = open("index" + str(index_num) + ".txt", 'w+')
    for term in sorted(words.keys()):
        index.write("{}:{}\n".format(term, words[term]))

    index.close()


def write_doc_id(documents):
    doc_counts = pickle.load(open('doc_id_map.pkl', 'rb'))
    doc_counts.update(documents)
    pickle.dump(doc_counts, open('doc_id_map.pkl', 'wb'))


def reset_index():
    pickle.dump({}, open("doc_id_map.pkl", "wb+"))
    pickle.dump({}, open("term_data.pkl", "wb+"))



main()