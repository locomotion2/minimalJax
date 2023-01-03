import time

import links_and_nodes as ln


class LNRemote(ln.lnm_remote):
    def __init__(self, address: str, debug: bool = False):
        super().__init__(address, debug=debug)

    def on_output(self, name, output):
        # disable printing of process output
        pass

    def on_log_messages(self, msgs):
        # disable printing of log messages
        pass

    def on_obj_state(self, obj_type, name, state):
        # Disable
        pass

    def start_process(self, process_name: str):
        self.request("set_process_state_request", ptype="Process", pname=process_name, requested_state="start")

    def stop_process(self, process_name: str):
        self.request("set_process_state_request", ptype="Process", pname=process_name, requested_state="stop")

    def restart_observers(self):
        self.stop_process("all/observers/distance")
        self.stop_process("all/observers/energy")
        time.sleep(1.0)
        print("Observers stopped")
        self.start_process("all/observers/distance")
        self.start_process("all/observers/energy")
        time.sleep(1.0)
        print("Observers restarted")
