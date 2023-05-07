CiteSeer_dprogram = '''
document(doc_1).
document(doc_2).

document_label(X1,X2,Y1,Y2,0) :- class(X1,Y1), class(X2,Y2).
document_label(X1,X2,Y1,Y2,1) :- class(X1,Y1), class(X2,Y2).

document_label(X1,X2,Y,Y,1) :- class(X1,Y), class(X2,Y).
document_label(X1,X2,Y,no_label,1) :- class(X1,Y), class(X2,Y).
document_label(X1,X2,no_label,Y,1) :- class(X1,Y), class(X2,Y).

class(empty,no_label).
class(X,Y) :- document_label_neural(0,X,Y).

nn(document_label_neural(1,X), [0,1,2,3,4,5]) :- document(X).
'''

Cora_dprogram = '''
document(doc_1).
document(doc_2).

document_label(X1,X2,Y1,Y2,0) :- class(X1,Y1), class(X2,Y2).
document_label(X1,X2,Y1,Y2,1) :- class(X1,Y1), class(X2,Y2).

document_label(X1,X2,Y,Y,1) :- class(X1,Y), class(X2,Y).
document_label(X1,X2,Y,no_label,1) :- class(X1,Y), class(X2,Y).
document_label(X1,X2,no_label,Y,1) :- class(X1,Y), class(X2,Y).

class(empty,no_label).
class(X,Y) :- document_label_neural(0,X,Y).

nn(document_label_neural(1,X), [0,1,2,3,4,5,6]) :- document(X).
'''

PubMed_dprogram = '''
document(doc_1).
document(doc_2).

document_label(X1,X2,Y1,Y2,0) :- class(X1,Y1), class(X2,Y2).
document_label(X1,X2,Y1,Y2,1) :- class(X1,Y1), class(X2,Y2).

document_label(X1,X2,Y,Y,1) :- class(X1,Y), class(X2,Y).
document_label(X1,X2,Y,no_label,1) :- class(X1,Y), class(X2,Y).
document_label(X1,X2,no_label,Y,1) :- class(X1,Y), class(X2,Y).

class(empty,no_label).
class(X,Y) :- document_label_neural(0,X,Y).

nn(document_label_neural(1,X), [0,1,2]) :- document(X).
'''

CiteSeer_dprogram_only_neural_network = '''
document_label(X,Y) :- document_label_neural(0,X,Y).
nn(document_label_neural(1,document), [0,1,2,3,4,5]).
'''
Cora_dprogram_only_neural_network = '''
document_label(X,Y) :- document_label_neural(0,X,Y).
nn(document_label_neural(1,document), [0,1,2,3,4,5,6]).
'''

PubMed_dprogram_only_neural_network = '''
document_label(X,Y) :- document_label_neural(0,X,Y).
nn(document_label_neural(1,document), [0,1,2]).
'''

CiteSeer_dprogram_original = '''
nn(document_label_neural(1,X), [0,1,2,3,4,5]) :- doc(X).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,0) :- cite(X,Z), document_label(Z,Y,1).
document_label(X,Y,0) :- document_label_neural(0,X,Y).
document_label(X,Y,1) :- document_label_neural(0,X,Y).
'''

Cora_dprogram_original = '''
nn(document_label_neural(1,X), [0,1,2,3,4,5,6]) :- doc(X).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,0) :- cite(X,Z), document_label(Z,Y,1).
document_label(X,Y,0) :- document_label_neural(0,X,Y).
document_label(X,Y,1) :- document_label_neural(0,X,Y).
'''

PubMed_dprogram_original = '''
nn(document_label_neural(1,X), [0,1,2]) :- doc(X).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,0) :- cite(X,Z), document_label(Z,Y,1).
document_label(X,Y,0) :- document_label_neural(0,X,Y).
document_label(X,Y,1) :- document_label_neural(0,X,Y).
'''