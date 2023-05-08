nn(document_net,[tensor(citeseer(X))],Y,[0,1,2,3,4,5]) :: document_label_neural(X,Y).

document_label(X,Y) :- document_label(X,Y,0).
document_label(X,Y,D) :- D < 1, cite(X,Z), D2 is D + 1, document_label(Z,Y,D2).
document_label(X,Y,1) :- document_label_neural(X,Y).

cite(X,Y) :- member(Y, [tensor(connected(X))]).