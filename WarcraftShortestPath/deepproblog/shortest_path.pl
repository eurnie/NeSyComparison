nn(tile_net,[I[X][Y]],Z,[0.8,1.2,5.3,7.7,9.2]) :: edge(I,X,Y,Z).

path(A,B) :- walk(A,B,[]).     

walk(A,B,V) :- edge(A,X),not(member(X,V)),(B = X;walk(X,B,[A|V])).

?- aggregate(min(D), path(a,g,D), D).
D = 8.



path(X1,Y1,X2,Y2) :- edge(_,X1,X2)


value_of(X, )

[1.2, 7.7, 9.2, 0.8, 5.3]

cost_of_path()

addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.

nn(citeseer_net,[tensor(citeseer(X))],Y,[0,1,2,3,4,5]) :: document_label_neural(X,Y).

document_label(X,Y) :- document_label_(X,Y,0).
document_label_(X,Y,D) :- D < 1, D2 is D + 1, cite(Z,X), document_label_(Z,Y,D2).
document_label_(X,Y,_) :- document_label_neural(X,Y).
