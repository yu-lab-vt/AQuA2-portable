from pathlib import Path
import re,unittest
from aqua2.project import Project,new_project


class EnglishInterface(unittest.TestCase):
    def test_application_source_has_no_chinese_text(self):
        root=Path(__file__).parents[1]/'src/aqua2'
        for path in root.glob('*.py'):
            self.assertIsNone(re.search(r'[\u3400-\u9fff]',path.read_text(encoding='utf-8')),path.name)

    def test_legacy_annotation_keys_are_merged_without_mutating_input(self):
        data=new_project('legacy');old='\u4e8b\u4ef6:channel1:local'
        data['annotations']['favorites']={old:[2,3],'Event:channel1:local':[1,2]}
        data['annotations']['hidden']={old:[3]}
        project=Project(data)
        self.assertEqual(project.data['annotations']['favorites'],{'Event:channel1:local':[1,2,3]})
        self.assertEqual(project.data['annotations']['hidden'],{'Event:channel1:local':[3]})
        self.assertIn(old,data['annotations']['favorites'])
