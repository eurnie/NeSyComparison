# tensor(train_mnist(0))

# KeyError: citeseer_documents(tensor(train(0)))

nn(citeseer_net,[tensor(citeseer(X))],Y,[0,1,2,3,4,5]) :: label(X,Y).
label(X,Y) :- linked(Z,X), label(Z,Y).