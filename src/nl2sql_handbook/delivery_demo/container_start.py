"""容器启动仅生成合成数据，再启动已安装的服务。"""
from nl2sql_handbook.evaluation_demo.dataset import prepare
from .server import main
if __name__ == "__main__":
    prepare()
    main(["--config", "/app/configs/service.docker.json"])
