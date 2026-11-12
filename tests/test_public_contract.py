"""Portable release checks; no AstrBot imports, credentials, or network calls."""
import ast
import json
from pathlib import Path
import re
import struct
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'main.py').read_text(encoding='utf-8')
SCHEMA = json.loads((ROOT / '_conf_schema.json').read_text(encoding='utf-8'))
META = (ROOT / 'metadata.yaml').read_text(encoding='utf-8')


class PublicReleaseTests(unittest.TestCase):
    def test_metadata_matches_runtime(self):
        tree = ast.parse(SOURCE)
        plugin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPTImage2Plugin')
        register = next(n for n in plugin.decorator_list if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'register')
        version = ast.literal_eval(register.args[3])
        self.assertIn('version: ' + version, META)
        self.assertIn('name: astrbot_plugin_gpt_image_2', META)
        self.assertIn('display_name: GPT Image · 龙绘', META)
        author = ast.literal_eval(register.args[1])
        self.assertEqual(author, 'aaron6769')
        self.assertIn('author: ' + author, META.splitlines())
        self.assertIn('https://github.com/aaron6769/astrbot-plugin-Image2-Studio', META)

    def test_native_models_have_no_private_defaults(self):
        api = SCHEMA['api']['items']
        for key in ('normal_provider_id', 'advanced_provider_id'):
            self.assertEqual(api[key]['_special'], 'select_provider')
            self.assertEqual(api[key]['default'], '')
            self.assertNotIn('options', api[key])
        self.assertFalse({'api_key','base_url','provider_id','model','advanced_model'} & api.keys())

    def test_conservative_gif_defaults(self):
        gif = SCHEMA['gif']['items']
        self.assertFalse(gif['enabled']['default'])
        self.assertEqual(gif['frame_count']['default'], 9)
        self.assertEqual(gif['frame_count']['options'], list(range(2, 17)))
        self.assertEqual(gif['frame_duration_ms']['default'], 200)
        self.assertEqual(gif['size']['default'], 'auto')
        self.assertEqual(SCHEMA['image']['items']['output_format']['default'], 'png')
        for key in ('frame_count','frame_duration_ms','size'):
            self.assertEqual(gif[key]['condition'], {'enabled':True})

    def test_dropdown_defaults_and_labels_are_valid(self):
        def walk(fields):
            for name, field in fields.items():
                if field['type'] == 'object':
                    walk(field['items'])
                if 'options' in field:
                    self.assertIn(field['default'], field['options'], name)
                    if 'labels' in field:
                        self.assertEqual(len(field['labels']),len(field['options']),name)
        walk(SCHEMA)

    def test_removed_persona_and_implicit_reference_settings_stay_removed(self):
        self.assertNotIn('self_persona', SOURCE)
        self.assertNotIn('self_persona', SCHEMA)
        self.assertNotIn('recent_reference_ttl_seconds', SCHEMA['runtime']['items'])
        self.assertIn('generate_image_with_gpt_image_2', SOURCE)

    def test_public_source_has_no_private_ipv4_endpoints(self):
        self.assertNotRegex(SOURCE, r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b')

    def test_compat_profile_selection_without_local_deployment_hint(self):
        # Compile only the unchanged selector methods, without importing AstrBot.
        tree = ast.parse(SOURCE)
        plugin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPTImage2Plugin')
        names = {'_base_url_prefers_standard_profile', '_auto_detect_profile', '_request_has_strong_profile_hint'}
        methods = [n for n in plugin.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(methods), len(names))
        namespace = {}
        exec(compile(ast.Module(body=methods, type_ignores=[]), 'profile-methods', 'exec'), namespace)

        class ProfileHarness:
            def _base_url(self):
                return 'https://images.example.invalid/v1'

            def _normalize_pixel_size(self, size):
                return size if size and re.fullmatch(r'\d+x\d+', size) else None

            def _parse_aspect_ratio(self, value):
                return value if value and re.fullmatch(r'\d+:\d+', value) else None

        for name in names:
            setattr(ProfileHarness, name, namespace[name])
        subject = ProfileHarness()
        for endpoint in ('https://ruoli.dev/v1', 'http://localhost:3000/v1', 'http://127.0.0.1:3000/v1'):
            self.assertTrue(subject._base_url_prefers_standard_profile(endpoint))
        self.assertFalse(subject._base_url_prefers_standard_profile())
        self.assertFalse(subject._base_url_prefers_standard_profile('http://192.0.2.10:3000/v1'))
        cases = [
            ('generate', None, None, None, 'standard', False),
            ('generate', '1024x1024', '16:9', None, 'standard', True),
            ('generate', None, '16:9', None, 'flexible', True),
            ('generate', None, '1:1', None, 'standard', False),
            ('edit', None, None, ['reference.png'], 'flexible', True),
        ]
        for operation, size, ratio, refs, profile, strong_hint in cases:
            with self.subTest(operation=operation, size=size, ratio=ratio):
                kwargs = dict(operation=operation, size=size, aspect_ratio=ratio, reference_image_paths=refs)
                self.assertEqual(subject._auto_detect_profile(**kwargs), profile)
                self.assertEqual(subject._request_has_strong_profile_hint(**kwargs), strong_hint)

    def test_logo_is_valid_square_png(self):
        data = (ROOT / 'logo.png').read_bytes()
        self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
        offset = 8
        seen_end = False
        while offset < len(data):
            size = struct.unpack('>I', data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            content = data[offset+8:offset+8+size]
            crc = struct.unpack('>I',data[offset+8+size:offset+12+size])[0]
            self.assertEqual(zlib.crc32(kind+content)&0xffffffff, crc)
            if kind == b'IHDR':
                width,height = struct.unpack('>II',content[:8])
                self.assertEqual(width,height)
                self.assertGreaterEqual(width,256)
            offset += 12 + size
            if kind == b'IEND':
                seen_end = True
                break
        self.assertTrue(seen_end)
        self.assertEqual(offset,len(data))

    def test_documentation_links_and_examples(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', readme):
            if '://' not in target and not target.startswith('#'):
                self.assertTrue((ROOT / target).is_file(), target)
        self.assertIn('src="logo.png"', readme)
        for command in ('/生图','/改图','/绘图','/plan confirm','/生图流程','--frames','--frame-ms'):
            self.assertIn(command, readme)

    def test_attribution_and_dependencies_are_present(self):
        self.assertIn('starmiaoa', (ROOT / 'LICENSE').read_text(encoding='utf-8'))
        artwork = (ROOT / 'ARTWORK_NOTICE.md').read_text(encoding='utf-8')
        self.assertIn('ZipZipPipe', artwork)
        self.assertIn('不可商用', artwork)
        requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8')
        self.assertIn('aiohttp', requirements)
        self.assertIn('Pillow', requirements)


if __name__ == '__main__':
    unittest.main()
