import mitsuba as mi

from emitters.vMF import VonMisesFisherEmitter


def register_emitters():
    mi.register_emitter('vMF', lambda props: VonMisesFisherEmitter(props))
