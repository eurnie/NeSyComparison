import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')



def main(lr,seed,perc_soft,l2w):
    num_examples = 50

    (x_train, hb_train), (x_test, hb_test) = datasets.mnist_follows(num_examples, seed=0, perc_soft = perc_soft)

    m_e = np.zeros_like(hb_train)
    m_e[:, num_examples*10:] = 1

    y_e_train = hb_train * m_e
    y_e_test = hb_test * m_e

    print(m_e)
    print(y_e_test)

    print(sum(m_e[0])/len(m_e[0]))

    """Logic Program Definition"""
    o = mme.Ontology()

    #Domains
    images = mme.Domain("Images", data=x_train)
    numbers = mme.Domain("Numbers", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T)
    result = mme.Domain("Result", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).T)
    o.add_domain([images, numbers, result])

    # # Predicates
    # digit = mme.Predicate("digit", domains=[images, numbers])
    # links = mme.Predicate("links", domains=[images, images], given=True)
    # follows = mme.Predicate("follows", domains=[numbers, numbers], given=True)
    # o.add_predicate([digit, links, follows])

    # Predicates
    digit = mme.Predicate("digit", domains=[images, numbers])
    images_sum = mme.Predicate("images_sum", domains=[images, images, result])
    digits_sum = mme.Predicate("digits_sum", domains=[numbers, numbers, result])
    o.add_predicate([digit, images_sum, digits_sum])

    """MME definition"""
    #Supervision
    indices = np.reshape(np.arange(images.num_constants * numbers.num_constants * result.num_constants),
                            [images.num_constants, numbers.num_constants, result.num_constants])
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(784,)))
    nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(10,use_bias=False))
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)

    #Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    p2 = mme.potentials.MutualExclusivityPotential(indices=indices)

    #Logical
    c = mme.Formula(definition="digit(x,i) and digit(y,j) and digits_sum(i,j,z) -> images_sum(x, y, z)", ontology=o)
    # c = mme.Formula(definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)", ontology=o)
    p3 = mme.potentials.EvidenceLogicPotential(formula=c,
                                                logic=mme.logic.BooleanLogic,
                                                evidence=y_e_train,
                                                evidence_mask=m_e)
    P = mme.potentials.GlobalPotential([p1,p2,p3])
    # pwt = mme.PieceWiseTraining(global_potential=P, y=hb_train)
    # pwt.compute_beta_logical_potentials()

    print("START TRAINING!!!")

    adam = tf.keras.optimizers.Adam(lr=0.001)

    def training_step():
        with tf.GradientTape() as tape:
            neural_logits = nn(x_train)

            total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_train,
                                                        logits=neural_logits)) + tf.reduce_sum(nn.losses)

        grads = tape.gradient(target=total_loss, sources=nn.variables)
        grad_vars = zip(grads, nn.variables)
        adam.apply_gradients(grad_vars)

    epochs = 150
    y_test = tf.reshape(hb_test[0, :num_examples * 10], [num_examples, 10])
    for _ in range(epochs):
        print("@@@@@@@@@@@@@@@@@@@@@")
        print(len(x_train))
        print(hb_train[0])
        training_step()
        # pwt.maximize_likelihood_step(hb_train, x=x_train)
        y_nn = p1.model(x_test)
        acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        # print(acc_nn)

    """Inference"""
    steps_map = 500
    hb = hb_test
    x = x_test
    evidence = y_e_test
    evidence_mask = m_e>0

    P.potentials[0].beta = 0.01
    map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                    potential=P,
                                                    logic=mme.logic.LukasiewiczLogic,
                                                    evidence=evidence,
                                                    evidence_mask=evidence_mask,
                                                    learning_rate= lr) #tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps_map, decay_rate=0.96, staircase=True))

    y_test = tf.reshape(hb[0, :num_examples * 10], [num_examples, 10])
    for i in range(steps_map):
        map_inference.infer_step(x)
        if i % 10 == 0:
            y_map = tf.reshape(map_inference.map()[0, :num_examples * 10], [num_examples, 10])
            acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
            # print("Accuracy MAP", acc_map.numpy())
            # print(y_map[:3])
        if mme.utils.heardEnter():
            break

    y_map = tf.reshape(map_inference.map()[0, :num_examples * 10], [num_examples, 10])
    y_nn = p1.model(x)

    acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))

    return [acc_map, acc_nn]

seed = 0

# setting the seeds
np.random.seed(seed)
tf.random.set_seed(seed)

res = []
possible_perc = [0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
possible_lr = [0.1]

perc = possible_perc[0]
lr = possible_lr[0]

acc_map, acc_nn = main(lr=lr, seed=seed, perc_soft=perc, l2w=0.01)
acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()

print("acc_map", acc_map)
print("acc_nn", acc_nn)


# res.append("\t".join([str(a) for a in [perc, lr, acc_map, str(acc_nn)+"\n"]]))
# for i in res:
    # print(i)

# with open("res_dlm_%d"%seed, "w") as file:
#     file.write("perc, lr, acc_map, acc_nn\n")
#     file.writelines(res)