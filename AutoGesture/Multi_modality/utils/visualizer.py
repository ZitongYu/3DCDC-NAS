#coding: utf8

import numpy as np
import time


class Visualizer():
    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='defult', **kwargs):
        '''
        ??visdom???
        '''
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d, modality):
        colmu_stac = []
        for k, v in d.items():
            colmu_stac.append(np.array(v))
        x = self.index.get(modality, 0)
        # self.vis.line(Y=np.column_stack((np.array(dicts['loss1']), np.array(dicts['loss2']))),
        self.vis.line(Y=np.column_stack(tuple(colmu_stac)),
                        X=np.array([x]),
                        win=(modality),
                        # opts=dict(title=modality,legend=['loss1', 'loss2'], ylabel='loss value'),
                      opts=dict(title=modality, legend=list(d.keys()), ylabel='Value', xlabel='Iteration'),
                      update=None if x == 0 else 'append')
        self.index[modality] = x + 1

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m.%d %H:%M:%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def img_grid(self, name, input_3d):
        """
        ??batch???????????i.e. input?36?64?64?
        ??? 6*6 ???????????64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def __getattr__(self, name):
        return getattr(self.vis, name)
