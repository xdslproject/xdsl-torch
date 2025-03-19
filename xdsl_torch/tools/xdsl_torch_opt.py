from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl_torch.dialects import get_all_dialects


class TorchOptMain(xDSLOptMain):
    def register_all_dialects(self):
        for name, dialect in get_all_dialects().items():
            self.ctx.register_dialect(name, dialect)


def main():
    TorchOptMain().run()


if __name__ == "__main__":
    main()
