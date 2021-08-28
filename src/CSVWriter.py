class CSVWriter:
    @staticmethod
    def write_dqn_stat_scv(filename, basic_dqn_stat, updated_dqn_stat):
        data_len = min(len(basic_dqn_stat), len(updated_dqn_stat))
        with open(filename, "w") as f:
            for i in range(data_len):
                f.write(";".join((str(i+1), str(basic_dqn_stat[i]), str(updated_dqn_stat))))