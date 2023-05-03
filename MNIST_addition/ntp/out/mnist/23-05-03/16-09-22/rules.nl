(('p0', 'X0', 'X1'), ('p1', 'X0', 'X2'), ('p2', 'X2', 'X1'))
0.0019911209	two(X,Y) :- seventeen(X,Z), seven(Z,Y).
0.0018913086	seven(X,Y) :- zero(X,Z), zero(Z,Y).
0.0017928814	fourteen(X,Y) :- two(X,Z), seventeen(Z,Y).
0.0017926395	fourteen(X,Y) :- fourteen(X,Z), two(Z,Y).
0.0017701535	seventeen(X,Y) :- two(X,Z), two(Z,Y).
0.0017081995	fourteen(X,Y) :- seventeen(X,Z), fourteen(Z,Y).
0.0016865588	fourteen(X,Y) :- two(X,Z), seventeen(Z,Y).
0.0016473302	seventeen(X,Y) :- two(X,Z), seven(Z,Y).
0.0016360827	zero(X,Y) :- fourteen(X,Z), fourteen(Z,Y).
0.0016249792	seven(X,Y) :- fourteen(X,Z), fourteen(Z,Y).
0.00152271	seventeen(X,Y) :- fourteen(X,Z), seven(Z,Y).
0.0014672773	seventeen(X,Y) :- two(X,Z), six(Z,Y).
0.0014607661	fourteen(X,Y) :- zero(X,Z), two(Z,Y).
0.0013897402	fourteen(X,Y) :- zero(X,Z), seven(Z,Y).
0.0012489454	one(X,Y) :- seventeen(X,Z), two(Z,Y).
0.0012102879	two(X,Y) :- seven(X,Z), one(Z,Y).
0.001066185	ten(X,Y) :- two(X,Z), seventeen(Z,Y).
0.0010100007	seventeen(X,Y) :- zero(X,Z), fourteen(Z,Y).
0.00085234176	fourteen(X,Y) :- fourteen(X,Z), seven(Z,Y).
0.00072292204	fourteen(X,Y) :- seven(X,Z), seven(Z,Y).

(('p0', 'X0', 'X1'), ('p1', 'X0', 'c'), ('p1', 'X1', 'c'))
0.9990005	nine(X,Y) :- digit(X,d3), digit(Y,d6).
0.9990005	nine(X,Y) :- digit(X,d6), digit(Y,d3).
0.9990005	twelve(X,Y) :- digit(X,d3), digit(Y,d9).
0.9990005	twelve(X,Y) :- digit(X,d6), digit(Y,d6).
0.9990005	twelve(X,Y) :- digit(X,d9), digit(Y,d3).
0.9990005	fifteen(X,Y) :- digit(X,d6), digit(Y,d9).
0.9990005	fifteen(X,Y) :- digit(X,d9), digit(Y,d6).
0.9980488	three(X,Y) :- digit(X,d0), digit(Y,d3).
0.9980488	three(X,Y) :- digit(X,d3), digit(Y,d0).
0.9980488	six(X,Y) :- digit(X,d0), digit(Y,d6).
0.9980488	six(X,Y) :- digit(X,d3), digit(Y,d3).
0.9980488	six(X,Y) :- digit(X,d6), digit(Y,d0).
0.9980488	nine(X,Y) :- digit(X,d0), digit(Y,d9).
0.9980488	nine(X,Y) :- digit(X,d9), digit(Y,d0).
0.9980488	eighteen(X,Y) :- digit(X,d9), digit(Y,d9).
0.9972417	two(X,Y) :- digit(X,d1), digit(Y,d1).
0.9972417	four(X,Y) :- digit(X,d0), digit(Y,d4).
0.9972417	four(X,Y) :- digit(X,d1), digit(Y,d3).
0.9972417	four(X,Y) :- digit(X,d3), digit(Y,d1).
0.9972417	four(X,Y) :- digit(X,d4), digit(Y,d0).
0.9972417	five(X,Y) :- digit(X,d1), digit(Y,d4).
0.9972417	five(X,Y) :- digit(X,d4), digit(Y,d1).
0.9972417	seven(X,Y) :- digit(X,d0), digit(Y,d7).
0.9972417	seven(X,Y) :- digit(X,d1), digit(Y,d6).
0.9972417	seven(X,Y) :- digit(X,d3), digit(Y,d4).
0.9972417	seven(X,Y) :- digit(X,d4), digit(Y,d3).
0.9972417	seven(X,Y) :- digit(X,d6), digit(Y,d1).
0.9972417	seven(X,Y) :- digit(X,d7), digit(Y,d0).
0.9972417	eight(X,Y) :- digit(X,d0), digit(Y,d8).
0.9972417	eight(X,Y) :- digit(X,d1), digit(Y,d7).
0.9972417	eight(X,Y) :- digit(X,d4), digit(Y,d4).
0.9972417	eight(X,Y) :- digit(X,d7), digit(Y,d1).
0.9972417	eight(X,Y) :- digit(X,d8), digit(Y,d0).
0.9972417	nine(X,Y) :- digit(X,d1), digit(Y,d8).
0.9972417	nine(X,Y) :- digit(X,d8), digit(Y,d1).
0.9972417	ten(X,Y) :- digit(X,d1), digit(Y,d9).
0.9972417	ten(X,Y) :- digit(X,d3), digit(Y,d7).
0.9972417	ten(X,Y) :- digit(X,d4), digit(Y,d6).
0.9972417	ten(X,Y) :- digit(X,d6), digit(Y,d4).
0.9972417	ten(X,Y) :- digit(X,d7), digit(Y,d3).
0.9972417	ten(X,Y) :- digit(X,d9), digit(Y,d1).
0.9972417	eleven(X,Y) :- digit(X,d3), digit(Y,d8).
0.9972417	eleven(X,Y) :- digit(X,d4), digit(Y,d7).
0.9972417	eleven(X,Y) :- digit(X,d7), digit(Y,d4).
0.9972417	eleven(X,Y) :- digit(X,d8), digit(Y,d3).
0.9972417	twelve(X,Y) :- digit(X,d4), digit(Y,d8).
0.9972417	twelve(X,Y) :- digit(X,d8), digit(Y,d4).
0.9972417	thirteen(X,Y) :- digit(X,d4), digit(Y,d9).
0.9972417	thirteen(X,Y) :- digit(X,d6), digit(Y,d7).
0.9972417	thirteen(X,Y) :- digit(X,d7), digit(Y,d6).
0.9972417	thirteen(X,Y) :- digit(X,d9), digit(Y,d4).
0.9972417	fourteen(X,Y) :- digit(X,d6), digit(Y,d8).
0.9972417	fourteen(X,Y) :- digit(X,d7), digit(Y,d7).
0.9972417	fourteen(X,Y) :- digit(X,d8), digit(Y,d6).
0.9972417	fifteen(X,Y) :- digit(X,d7), digit(Y,d8).
0.9972417	fifteen(X,Y) :- digit(X,d8), digit(Y,d7).
0.9972417	sixteen(X,Y) :- digit(X,d7), digit(Y,d9).
0.9972417	sixteen(X,Y) :- digit(X,d8), digit(Y,d8).
0.9972417	sixteen(X,Y) :- digit(X,d9), digit(Y,d7).
0.9972417	seventeen(X,Y) :- digit(X,d8), digit(Y,d9).
0.9972417	seventeen(X,Y) :- digit(X,d9), digit(Y,d8).
0.9966228	zero(X,Y) :- digit(X,d0), digit(Y,d0).
0.9966228	one(X,Y) :- digit(X,d0), digit(Y,d1).
0.9966228	one(X,Y) :- digit(X,d1), digit(Y,d0).
0.9966228	two(X,Y) :- digit(X,d0), digit(Y,d2).
0.9966228	two(X,Y) :- digit(X,d2), digit(Y,d0).
0.9966228	three(X,Y) :- digit(X,d1), digit(Y,d2).
0.9966228	three(X,Y) :- digit(X,d2), digit(Y,d1).
0.9966228	four(X,Y) :- digit(X,d2), digit(Y,d2).
0.9966228	five(X,Y) :- digit(X,d2), digit(Y,d3).
0.9966228	five(X,Y) :- digit(X,d3), digit(Y,d2).
0.9966228	six(X,Y) :- digit(X,d2), digit(Y,d4).
0.9966228	six(X,Y) :- digit(X,d4), digit(Y,d2).
0.9966228	eight(X,Y) :- digit(X,d2), digit(Y,d6).
0.9966228	eight(X,Y) :- digit(X,d6), digit(Y,d2).
0.9966228	nine(X,Y) :- digit(X,d2), digit(Y,d7).
0.9966228	nine(X,Y) :- digit(X,d7), digit(Y,d2).
0.9966228	ten(X,Y) :- digit(X,d2), digit(Y,d8).
0.9966228	ten(X,Y) :- digit(X,d8), digit(Y,d2).
0.9966228	eleven(X,Y) :- digit(X,d2), digit(Y,d9).
0.9966228	eleven(X,Y) :- digit(X,d9), digit(Y,d2).
0.9961014	five(X,Y) :- digit(X,d0), digit(Y,d5).
0.9961014	five(X,Y) :- digit(X,d5), digit(Y,d0).
0.9961014	six(X,Y) :- digit(X,d1), digit(Y,d5).
0.9961014	six(X,Y) :- digit(X,d5), digit(Y,d1).
0.9961014	seven(X,Y) :- digit(X,d2), digit(Y,d5).
0.9961014	seven(X,Y) :- digit(X,d5), digit(Y,d2).
0.9961014	eight(X,Y) :- digit(X,d3), digit(Y,d5).
0.9961014	eight(X,Y) :- digit(X,d5), digit(Y,d3).
0.9961014	nine(X,Y) :- digit(X,d4), digit(Y,d5).
0.9961014	nine(X,Y) :- digit(X,d5), digit(Y,d4).
0.9961014	ten(X,Y) :- digit(X,d5), digit(Y,d5).
0.9961014	eleven(X,Y) :- digit(X,d5), digit(Y,d6).
0.9961014	eleven(X,Y) :- digit(X,d6), digit(Y,d5).
0.9961014	twelve(X,Y) :- digit(X,d5), digit(Y,d7).
0.9961014	twelve(X,Y) :- digit(X,d7), digit(Y,d5).
0.9961014	thirteen(X,Y) :- digit(X,d5), digit(Y,d8).
0.9961014	thirteen(X,Y) :- digit(X,d8), digit(Y,d5).
0.9961014	fourteen(X,Y) :- digit(X,d5), digit(Y,d9).
0.9961014	fourteen(X,Y) :- digit(X,d9), digit(Y,d5).

(('p0', 'X0', 'X1'), ('p1', 'X1', 'X0'))
0.0025574423	two(X,Y) :- fourteen(Y,X).
0.0021515756	fourteen(X,Y) :- seven(Y,X).
0.002134834	two(X,Y) :- seventeen(Y,X).
0.0019957623	two(X,Y) :- digit(Y,X).
0.0019893567	zero(X,Y) :- nine(Y,X).
0.0019706206	seventeen(X,Y) :- seventeen(Y,X).
0.0018005853	two(X,Y) :- two(Y,X).
0.0017747183	seven(X,Y) :- two(Y,X).
0.0017190967	zero(X,Y) :- zero(Y,X).
0.0016988483	eight(X,Y) :- two(Y,X).
0.001647958	fourteen(X,Y) :- fourteen(Y,X).
0.0014718166	fourteen(X,Y) :- five(Y,X).
0.0014656542	two(X,Y) :- digit(Y,X).
0.0014587378	seven(X,Y) :- two(Y,X).
0.0014138585	two(X,Y) :- two(Y,X).
0.0013741865	zero(X,Y) :- seven(Y,X).
0.0013094066	digit(X,Y) :- seven(Y,X).
0.0011178788	two(X,Y) :- two(Y,X).
0.0009422654	zero(X,Y) :- zero(Y,X).
0.0008254005	two(X,Y) :- two(Y,X).

