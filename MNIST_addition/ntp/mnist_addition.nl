zero(X,Y) :- digit(X,d0), digit(Y,d0).
one(X,Y) :- digit(X,d0), digit(Y,d1).
one(X,Y) :- digit(X,d1), digit(Y,d0).
two(X,Y) :- digit(X,d0), digit(Y,d2).
two(X,Y) :- digit(X,d1), digit(Y,d1).
two(X,Y) :- digit(X,d2), digit(Y,d0).
three(X,Y) :- digit(X,d0), digit(Y,d3).
three(X,Y) :- digit(X,d1), digit(Y,d2).
three(X,Y) :- digit(X,d2), digit(Y,d1).
three(X,Y) :- digit(X,d3), digit(Y,d0).
four(X,Y) :- digit(X,d0), digit(Y,d4).
four(X,Y) :- digit(X,d1), digit(Y,d3).
four(X,Y) :- digit(X,d2), digit(Y,d2).
four(X,Y) :- digit(X,d3), digit(Y,d1).
four(X,Y) :- digit(X,d4), digit(Y,d0).
five(X,Y) :- digit(X,d0), digit(Y,d5).
five(X,Y) :- digit(X,d1), digit(Y,d4).
five(X,Y) :- digit(X,d2), digit(Y,d3).
five(X,Y) :- digit(X,d3), digit(Y,d2).
five(X,Y) :- digit(X,d4), digit(Y,d1).
five(X,Y) :- digit(X,d5), digit(Y,d0).
six(X,Y) :- digit(X,d0), digit(Y,d6).
six(X,Y) :- digit(X,d1), digit(Y,d5).
six(X,Y) :- digit(X,d2), digit(Y,d4).
six(X,Y) :- digit(X,d3), digit(Y,d3).
six(X,Y) :- digit(X,d4), digit(Y,d2).
six(X,Y) :- digit(X,d5), digit(Y,d1).
six(X,Y) :- digit(X,d6), digit(Y,d0).
seven(X,Y) :- digit(X,d0), digit(Y,d7).
seven(X,Y) :- digit(X,d1), digit(Y,d6).
seven(X,Y) :- digit(X,d2), digit(Y,d5).
seven(X,Y) :- digit(X,d3), digit(Y,d4).
seven(X,Y) :- digit(X,d4), digit(Y,d3).
seven(X,Y) :- digit(X,d5), digit(Y,d2).
seven(X,Y) :- digit(X,d6), digit(Y,d1).
seven(X,Y) :- digit(X,d7), digit(Y,d0).
eight(X,Y) :- digit(X,d0), digit(Y,d8).
eight(X,Y) :- digit(X,d1), digit(Y,d7).
eight(X,Y) :- digit(X,d2), digit(Y,d6).
eight(X,Y) :- digit(X,d3), digit(Y,d5).
eight(X,Y) :- digit(X,d4), digit(Y,d4).
eight(X,Y) :- digit(X,d5), digit(Y,d3).
eight(X,Y) :- digit(X,d6), digit(Y,d2).
eight(X,Y) :- digit(X,d7), digit(Y,d1).
eight(X,Y) :- digit(X,d8), digit(Y,d0).
nine(X,Y) :- digit(X,d0), digit(Y,d9).
nine(X,Y) :- digit(X,d1), digit(Y,d8).
nine(X,Y) :- digit(X,d2), digit(Y,d7).
nine(X,Y) :- digit(X,d3), digit(Y,d6).
nine(X,Y) :- digit(X,d4), digit(Y,d5).
nine(X,Y) :- digit(X,d5), digit(Y,d4).
nine(X,Y) :- digit(X,d6), digit(Y,d3).
nine(X,Y) :- digit(X,d7), digit(Y,d2).
nine(X,Y) :- digit(X,d8), digit(Y,d1).
nine(X,Y) :- digit(X,d9), digit(Y,d0).
ten(X,Y) :- digit(X,d1), digit(Y,d9).
ten(X,Y) :- digit(X,d2), digit(Y,d8).
ten(X,Y) :- digit(X,d3), digit(Y,d7).
ten(X,Y) :- digit(X,d4), digit(Y,d6).
ten(X,Y) :- digit(X,d5), digit(Y,d5).
ten(X,Y) :- digit(X,d6), digit(Y,d4).
ten(X,Y) :- digit(X,d7), digit(Y,d3).
ten(X,Y) :- digit(X,d8), digit(Y,d2).
ten(X,Y) :- digit(X,d9), digit(Y,d1).
eleven(X,Y) :- digit(X,d2), digit(Y,d9).
eleven(X,Y) :- digit(X,d3), digit(Y,d8).
eleven(X,Y) :- digit(X,d4), digit(Y,d7).
eleven(X,Y) :- digit(X,d5), digit(Y,d6).
eleven(X,Y) :- digit(X,d6), digit(Y,d5).
eleven(X,Y) :- digit(X,d7), digit(Y,d4).
eleven(X,Y) :- digit(X,d8), digit(Y,d3).
eleven(X,Y) :- digit(X,d9), digit(Y,d2).
twelve(X,Y) :- digit(X,d3), digit(Y,d9).
twelve(X,Y) :- digit(X,d4), digit(Y,d8).
twelve(X,Y) :- digit(X,d5), digit(Y,d7).
twelve(X,Y) :- digit(X,d6), digit(Y,d6).
twelve(X,Y) :- digit(X,d7), digit(Y,d5).
twelve(X,Y) :- digit(X,d8), digit(Y,d4).
twelve(X,Y) :- digit(X,d9), digit(Y,d3).
thirteen(X,Y) :- digit(X,d4), digit(Y,d9).
thirteen(X,Y) :- digit(X,d5), digit(Y,d8).
thirteen(X,Y) :- digit(X,d6), digit(Y,d7).
thirteen(X,Y) :- digit(X,d7), digit(Y,d6).
thirteen(X,Y) :- digit(X,d8), digit(Y,d5).
thirteen(X,Y) :- digit(X,d9), digit(Y,d4).
fourteen(X,Y) :- digit(X,d5), digit(Y,d9).
fourteen(X,Y) :- digit(X,d6), digit(Y,d8).
fourteen(X,Y) :- digit(X,d7), digit(Y,d7).
fourteen(X,Y) :- digit(X,d8), digit(Y,d6).
fourteen(X,Y) :- digit(X,d9), digit(Y,d5).
fifteen(X,Y) :- digit(X,d6), digit(Y,d9).
fifteen(X,Y) :- digit(X,d7), digit(Y,d8).
fifteen(X,Y) :- digit(X,d8), digit(Y,d7).
fifteen(X,Y) :- digit(X,d9), digit(Y,d6).
sixteen(X,Y) :- digit(X,d7), digit(Y,d9).
sixteen(X,Y) :- digit(X,d8), digit(Y,d8).
sixteen(X,Y) :- digit(X,d9), digit(Y,d7).
seventeen(X,Y) :- digit(X,d8), digit(Y,d9).
seventeen(X,Y) :- digit(X,d9), digit(Y,d8).
eighteen(X,Y) :- digit(X,d9), digit(Y,d9).

six(train-52198,train-15910).
nine(train-1101,train-30043).
ten(train-5093,train-32756).
three(train-26038,train-41337).
nine(train-34679,train-28401).
nine(train-17616,train-40483).
four(train-59479,train-18804).
eight(train-47650,train-45013).
eleven(train-38021,train-38144).
thirteen(train-15092,train-7975).
eighteen(train-27074,train-29475).
nine(train-22077,train-57790).
twelve(train-37150,train-26542).
ten(train-56599,train-59114).
nine(train-31009,train-12860).
nine(train-30518,train-13703).
sixteen(train-22151,train-8753).
four(train-42304,train-9552).
four(train-9742,train-28290).
eight(train-3407,train-24440).
one(train-45900,train-1701).
eleven(train-11154,train-14492).
five(train-54184,train-51831).
six(train-24784,train-52487).
nine(train-58656,train-26377).
eight(train-50551,train-24724).
six(train-35348,train-6837).

unknown(train-24413,train-42772).
unknown(train-6584,train-51319).
unknown(train-6641,train-34757).

unknown(test-9770,test-3631).
unknown(test-7059,test-421).
unknown(test-9878,test-7213).
unknown(test-7847,test-9353).
unknown(test-6686,test-5617).

