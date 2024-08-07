import multiprocessing

import pytest
from xdist.scheduler import LoadGroupScheduling
import psutil


# def pytest_addoption(parser):
#     parser.addoption("--tensorflow-cpus", action="store", default=1,
#                      help="Number of CPUs for TensorFlow tests")
#     parser.addoption("--torch-cpus", action="store", default=1,
#                      help="Number of CPUs for PyTorch tests")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tensorflow: mark test to run with TensorFlow backend"
    )
    config.addinivalue_line("markers", "torch: mark test to run with PyTorch backend")
    # worker_id = os.environ.get("PYTEST_XDIST_WORKER")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    num_cpus = config.getoption("--numprocesses")
    if num_cpus is None or num_cpus == "auto":
        num_cpus = multiprocessing.cpu_count()
    elif num_cpus == "logical":
        num_cpus = psutil.cpu_count(logical=True)
    else:
        num_cpus = int(num_cpus)

    assert num_cpus > 1, "Num of CPUs cannot be 1"

    if num_cpus > 1:
        # cpus = {
        #     "tensorflow": int(config.getoption("--tensorflow-cpus")),
        #     "torch": int(config.getoption("--torch-cpus")),
        # }

        def check_keyword(kword, item):
            if kword in item.keywords:
                return True

            for keyword in item.keywords:
                if kword in keyword:
                    return True

            return False

        grouped_items = {}
        others = []
        for item in items:
            if check_keyword("tensorflow", item):
                key = "tensorflow"
            elif check_keyword("torch", item):
                key = "torch"
            else:
                others.append(item)
                continue

            setattr(item, "xfail", True)  # Mark as xfail initially

            item.add_marker(getattr(pytest.mark, key))
            item.add_marker(pytest.mark.xdist_group(key))
            # print(item.get_closest_marker("xdist_group"))

            grouped_items.setdefault(key, []).append(item)

        items.clear()
        for group in grouped_items.values():
            items.extend(group)
        items.extend(others)


class MyLoadGroupScheduling(LoadGroupScheduling):
    def _split_scope(self, nodeid):
        # TODO: How are the @group_name generated?
        if nodeid.endswith("tensorflow]"):
            return "tensorflow"

        if nodeid.endswith("torch]"):
            return "torch"

        return super()._split_scope(nodeid)


def pytest_xdist_make_scheduler(config, log):
    return MyLoadGroupScheduling(config, log)


# def pytest_xdist_setupnodes(config, specs):
#     num_cpus = config.getoption("--numprocesses")
#     if num_cpus is None or num_cpus == "auto":
#         num_cpus = multiprocessing.cpu_count()
#     elif num_cpus == "logical":
#         num_cpus = psutil.cpu_count(logical=True)
#     else:
#         num_cpus = int(num_cpus)

#     for spec in specs:
#         if spec.args is None:
#             spec.args = []
