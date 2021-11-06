import os

import pytest

SCRIPTS = [x for x in os.listdir("scripts/") if x[-3:] == ".py"]


class TestScripts:
    @pytest.mark.parametrize("script", SCRIPTS)
    def test_script(self, script: str, script_runner):
        ret = script_runner.run(
            os.getcwd() + "/scripts/" + script, cwd=os.getcwd() + "/scripts/"
        )

        assert ret.success
