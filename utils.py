import ray

if not ray.is_initialized():
    ray.init(dashboard_host='0.0.0.0')


def pmap(func, xs):
    rfunc = ray.remote(func)
    xs = [rfunc.remote(x) for x in xs]
    return ray.get(xs)
