import tensorflow as tf


def extend_with_gradient_accumulation_v2(BaseOptimizer):
    """返回新的优化器类，加入梯度累积
    """

    class NewOptimizer(BaseOptimizer):
        """带有梯度累积的优化器
        """

        def __init__(self, grad_accum_steps, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self.grad_accum_steps = grad_accum_steps

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'ag')

        def _resource_apply(self, grad, var, indices=None):
            # 更新判据
            cond = tf.keras.backend.equal(self.iterations % self.grad_accum_steps, 0)
            # 获取梯度
            ag = self.get_slot(var, 'ag')

            old_update = tf.keras.backend.update

            def new_update(x, new_x):
                new_x = tf.keras.backend.switch(cond, new_x, x)
                return old_update(x, new_x)

            tf.keras.backend.update = new_update
            ag_t = ag / self.grad_accum_steps
            op = super(NewOptimizer, self)._resource_apply(ag_t, var)
            tf.keras.backend.update = old_update

            # 累积梯度
            with tf.control_dependencies([op]):
                ag_t = tf.keras.backend.switch(cond, tf.keras.backend.zeros_like(ag), ag)
                with tf.control_dependencies([tf.keras.backend.update(ag, ag_t)]):
                    if indices is None:
                        ag_t = tf.keras.backend.update(ag, ag + grad)
                    else:
                        ag_t = self._resource_scatter_add(ag, indices, grad)

            return ag_t

        def get_config(self):
            config = {
                'grad_accum_steps': self.grad_accum_steps,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer
