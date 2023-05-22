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
seventeen(train-49165,train-19806).
eight(train-5230,train-43477).
eight(train-53301,train-4432).
ten(train-1752,train-55800).
eight(train-29633,train-35719).
seven(train-25921,train-29850).
eleven(train-35627,train-44816).
six(train-50182,train-11242).
one(train-33378,train-20899).
fourteen(train-50186,train-14630).
five(train-24238,train-55858).
four(train-56738,train-24341).
six(train-5820,train-20797).
one(train-20459,train-25823).
two(train-5256,train-52622).
nine(train-56119,train-49843).
twelve(train-6563,train-34185).
eleven(train-5822,train-15597).
fifteen(train-31160,train-41253).
thirteen(train-30513,train-35868).
twelve(train-28433,train-37176).
nine(train-12180,train-8276).
nine(train-57092,train-28676).
ten(train-8818,train-26734).
four(train-48951,train-29626).
one(train-49800,train-59773).
four(train-55008,train-1807).
eight(train-37122,train-37953).
ten(train-18853,train-8036).
eight(train-51980,train-3293).
seven(train-37926,train-33030).
ten(train-15874,train-13387).
six(train-34162,train-53479).
eight(train-7686,train-13504).
twelve(train-16678,train-54698).
twelve(train-26710,train-53752).
ten(train-10310,train-33073).
eleven(train-21039,train-38741).
eight(train-31529,train-23782).
six(train-21156,train-54016).
two(train-4269,train-30625).
twelve(train-9614,train-34597).
eleven(train-50765,train-52794).
nine(train-34103,train-29614).
five(train-29530,train-50437).
twelve(train-53935,train-17355).
eight(train-11411,train-26741).
fourteen(train-17283,train-39666).
seventeen(train-24679,train-19239).
five(train-3212,train-34781).
twelve(train-25378,train-42155).
twelve(train-18445,train-58273).
seven(train-45154,train-13657).
five(train-52810,train-49098).
eight(train-40599,train-35574).
six(train-47499,train-57630).
seventeen(train-45039,train-23391).
fourteen(train-28541,train-56052).
seven(train-2455,train-56968).
eight(train-14920,train-35048).
fifteen(train-54523,train-19437).
thirteen(train-16257,train-32061).
three(train-8841,train-14234).
eight(train-28897,train-27528).
eight(train-6718,train-16852).
nine(train-21105,train-3375).
six(train-1039,train-10006).
seven(train-21877,train-9108).
thirteen(train-17887,train-24785).
seventeen(train-29905,train-17393).
nine(train-4675,train-15865).
nine(train-39660,train-24480).
four(train-43252,train-57433).
three(train-50457,train-31174).
one(train-49697,train-37758).
ten(train-51977,train-10311).
seventeen(train-19570,train-3787).
eighteen(train-12443,train-1038).
eight(train-16652,train-4410).
thirteen(train-51445,train-27626).
eight(train-21425,train-15957).
nine(train-24385,train-49157).
thirteen(train-38685,train-53634).
seven(train-53350,train-57286).
four(train-8156,train-28132).
eight(train-8797,train-46286).
eleven(train-40183,train-18531).
twelve(train-33093,train-23671).
ten(train-11337,train-29868).
nine(train-53248,train-50621).
eleven(train-58100,train-59383).
eighteen(train-6564,train-43385).
five(train-27287,train-57649).
fifteen(train-16964,train-47677).
thirteen(train-35398,train-53794).
ten(train-26502,train-28047).
eleven(train-10130,train-53437).
nine(train-44508,train-40225).
six(train-41887,train-17402).
six(train-41029,train-10903).
seventeen(train-51612,train-14208).
five(train-36266,train-875).
eleven(train-53122,train-38875).
nine(train-1475,train-47119).
twelve(train-2501,train-45196).
ten(train-15550,train-42564).
eight(train-25243,train-48312).
thirteen(train-49851,train-19874).
thirteen(train-5411,train-8950).
nine(train-37782,train-4482).
seven(train-17174,train-26225).
sixteen(train-22967,train-8803).
ten(train-20099,train-14824).
fifteen(train-58168,train-8651).
eight(train-3151,train-14070).
five(train-34383,train-8325).
five(train-50372,train-28234).
eleven(train-51756,train-12609).
four(train-13831,train-1003).
three(train-7462,train-45272).
ten(train-12075,train-12728).
fifteen(train-26886,train-31721).
thirteen(train-10913,train-53291).
thirteen(train-1574,train-12036).
six(train-44334,train-44685).
seven(train-10926,train-4364).
eleven(train-17183,train-41173).
fourteen(train-1743,train-35727).
twelve(train-33805,train-53155).
ten(train-32503,train-36775).
eight(train-18052,train-39079).
seven(train-26326,train-32254).
fifteen(train-56932,train-48842).
eleven(train-1473,train-6397).
two(train-41694,train-1860).
nine(train-43482,train-39188).
seven(train-20088,train-3214).
six(train-56415,train-37003).
fourteen(train-36069,train-16915).
nine(train-37584,train-38568).
thirteen(train-54681,train-5179).
five(train-52813,train-37056).
seven(train-43281,train-3431).
eleven(train-19919,train-23853).
fourteen(train-54880,train-1292).
six(train-16551,train-14836).
ten(train-2091,train-36126).
nine(train-43218,train-37381).
eight(train-43341,train-2366).
ten(train-41981,train-32302).
four(train-25725,train-22768).
three(train-51724,train-23701).
eight(train-20729,train-43910).
four(train-42789,train-14363).
ten(train-47853,train-36257).
nine(train-2819,train-14195).
thirteen(train-28591,train-5821).
nine(train-33033,train-16443).
one(train-33219,train-35995).
fourteen(train-15178,train-21300).
four(train-18420,train-59878).
eleven(train-13606,train-20445).
seven(train-38662,train-22876).
seven(train-51425,train-49363).
six(train-30590,train-49192).
seventeen(train-55764,train-19728).
two(train-27665,train-19639).
three(train-43685,train-20570).
four(train-32246,train-56761).
seven(train-59273,train-35360).
ten(train-19350,train-8293).
fifteen(train-48477,train-59959).
six(train-36909,train-385).
six(train-20633,train-41169).
seven(train-5160,train-40946).
seven(train-12990,train-47413).
twelve(train-38178,train-56033).
eleven(train-40997,train-7614).
nine(train-42322,train-59931).
eight(train-28562,train-13334).
sixteen(train-50903,train-12903).
sixteen(train-13573,train-59388).
seven(train-14533,train-13008).
six(train-21458,train-33969).
eight(train-13876,train-16318).
six(train-55721,train-6954).
seven(train-47519,train-59144).
twelve(train-16488,train-48415).
seven(train-31731,train-28565).
thirteen(train-47988,train-53336).
eight(train-15098,train-10686).
five(train-21320,train-48034).
six(train-15788,train-34730).
eleven(train-50854,train-29699).
one(train-24389,train-32795).
five(train-17458,train-28478).
ten(train-33607,train-18537).
twelve(train-53061,train-16865).
twelve(train-10402,train-20792).
eight(train-55178,train-57499).
fourteen(train-57344,train-51532).
twelve(train-36246,train-54162).
three(train-40054,train-50222).
six(train-507,train-23123).
eight(train-46783,train-5026).
twelve(train-12265,train-14956).
thirteen(train-2934,train-3370).
six(train-11636,train-49816).
four(train-24279,train-45704).
ten(train-43749,train-5700).
four(train-43768,train-29176).
nine(train-47099,train-45204).
nine(train-18360,train-43698).
nine(train-42328,train-27561).
ten(train-56762,train-5078).
five(train-1680,train-43634).
three(train-15786,train-4341).
eleven(train-21570,train-39793).
six(train-57029,train-5697).
six(train-58479,train-33605).
thirteen(train-5489,train-5763).
seven(train-4586,train-33623).
nine(train-32137,train-53900).
eight(train-50670,train-26875).
eleven(train-32967,train-13198).
eleven(train-59040,train-44824).
four(train-33106,train-32336).
seventeen(train-7797,train-19567).
fifteen(train-37672,train-15081).
ten(train-29061,train-30060).
twelve(train-19,train-1449).
eight(train-41204,train-38555).
eleven(train-17239,train-44635).
ten(train-26746,train-46206).
nine(train-55043,train-20506).
twelve(train-33692,train-2422).
fourteen(train-25201,train-20583).
fourteen(train-57218,train-54367).
thirteen(train-21418,train-32247).
five(train-17854,train-22518).
two(train-29456,train-3161).
twelve(train-38844,train-45404).
twelve(train-27448,train-54767).

unknown(train-24413,train-42772).
unknown(train-6584,train-51319).
unknown(train-6641,train-34757).
unknown(train-4076,train-50822).
unknown(train-42060,train-7211).
unknown(train-37136,train-51406).
unknown(train-42596,train-44421).
unknown(train-43191,train-25615).
unknown(train-36357,train-23619).
unknown(train-29032,train-28191).
unknown(train-14807,train-48826).
unknown(train-14298,train-11129).
unknown(train-41612,train-20852).
unknown(train-45947,train-32866).
unknown(train-19273,train-51589).
unknown(train-19791,train-34742).
unknown(train-28507,train-20657).
unknown(train-38999,train-37649).
unknown(train-28340,train-58230).
unknown(train-35361,train-22662).
unknown(train-17601,train-14576).
unknown(train-20939,train-46879).
unknown(train-27886,train-24957).
unknown(train-45079,train-42747).
unknown(train-35170,train-59580).
unknown(train-41164,train-19658).
unknown(train-52736,train-42520).
unknown(train-4824,train-53238).
unknown(train-34172,train-9046).
unknown(train-41228,train-8687).

unknown(test-9770,test-3631).
unknown(test-7059,test-421).
unknown(test-9878,test-7213).
unknown(test-7847,test-9353).
unknown(test-6686,test-5617).
unknown(test-4179,test-5546).
unknown(test-4963,test-9111).
unknown(test-4775,test-2151).
unknown(test-7050,test-270).
unknown(test-4547,test-7812).
unknown(test-3723,test-7022).
unknown(test-2403,test-5846).
unknown(test-4004,test-7368).
unknown(test-308,test-8081).
unknown(test-507,test-3072).
unknown(test-2276,test-8585).
unknown(test-7811,test-3176).
unknown(test-5732,test-9192).
unknown(test-2673,test-6926).
unknown(test-7434,test-9953).
unknown(test-4669,test-1675).
unknown(test-31,test-4583).
unknown(test-477,test-3044).
unknown(test-5464,test-4538).
unknown(test-899,test-4944).
unknown(test-2380,test-4550).
unknown(test-1742,test-4092).
unknown(test-3090,test-1326).
unknown(test-2595,test-1728).
unknown(test-2189,test-7860).
unknown(test-7592,test-9452).
unknown(test-4687,test-9348).
unknown(test-4129,test-8044).
unknown(test-6666,test-1583).
unknown(test-4514,test-2373).
unknown(test-3645,test-6074).
unknown(test-8199,test-369).
unknown(test-7360,test-9687).
unknown(test-7760,test-6258).
unknown(test-1416,test-8936).
unknown(test-4302,test-2631).
unknown(test-5239,test-9651).
unknown(test-4019,test-1661).
unknown(test-3804,test-8560).
unknown(test-5311,test-4452).
unknown(test-1194,test-2385).
unknown(test-7937,test-8859).
unknown(test-7976,test-6488).
unknown(test-8748,test-1148).
unknown(test-8446,test-8362).

