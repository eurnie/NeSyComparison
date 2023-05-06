CiteSeer_dprogram_only_neural_network = '''
document_label(X,Y) :- document_label_neural(0,X,Y).
nn(document_label_neural(1,document), [0,1,2,3,4,5]).
'''
Cora_dprogram_only_neural_network = '''
document_label(X,Y) :- document_label_neural(0,X,Y).
nn(document_label_neural(1,document), [0,1,2,3,4,5]).
'''

PubMed_dprogram_only_neural_network = '''
document_label(X,Y) :- document_label_neural(0,X,Y).
nn(document_label_neural(1,document), [0,1,2,3,4,5]).
'''

CiteSeer_dprogram = '''
nn(document_label_neural(1,X), [0,1,2,3,4,5]) :- doc(X).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,0) :- cite(X,Z), document_label(Z,Y,1).
document_label(X,Y,0) :- document_label_neural(0,X,Y).
document_label(X,Y,1) :- document_label_neural(0,X,Y).
'''

Cora_dprogram = '''
nn(document_label_neural(1,X), [0,1,2,3,4,5,6]) :- doc(X).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,0) :- cite(X,Z), document_label(Z,Y,1).
document_label(X,Y,0) :- document_label_neural(0,X,Y).
document_label(X,Y,1) :- document_label_neural(0,X,Y).
'''

PubMed_dprogram = '''
nn(document_label_neural(1,X), [0,1,2]) :- doc(X).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,0) :- cite(X,Z), document_label(Z,Y,1).
document_label(X,Y,0) :- document_label_neural(0,X,Y).
document_label(X,Y,1) :- document_label_neural(0,X,Y).
'''