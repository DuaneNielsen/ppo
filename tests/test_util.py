from util import *


def test_converged():
    c = Converged(min_change=0.001)
    loss = 0.001
    change = 0.0001
    loss -= change
    assert c.converged(loss) == False
    loss -= change
    assert c.converged(loss) == False
    loss -= change
    assert c.converged(loss) == True

    c.reset()

    loss -= change
    assert c.converged(loss) == False
    loss -= change
    assert c.converged(loss) == False
    loss -= change
    assert c.converged(loss) == True


def test_converged2():
    c = Converged(min_change=0.001)
    loss = 0.001
    bigchange = 0.1
    smallchange = 0.0001
    loss -= bigchange
    assert c.converged(loss) == False
    loss -= bigchange
    assert c.converged(loss) == False
    loss -= bigchange
    assert c.converged(loss) == False

    loss -= smallchange
    assert c.converged(loss) == False
    loss -= smallchange
    assert c.converged(loss) == True

    loss -= bigchange
    assert c.converged(loss) == True
    loss -= smallchange
    assert c.converged(loss) == True


def test_converged_parameters():
    c = Converged(min_change=0.001, detections=5, detection_window=8)

    loss = 0.001
    bigchange = 0.1
    smallchange = 0.0001

    for _ in range(200):
        loss -= bigchange
        assert c.converged(loss) == False

    for _ in range(4):
        loss -= smallchange
        assert c.converged(loss) == False

    loss -= smallchange
    assert c.converged(loss) == True

