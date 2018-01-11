$PROJECT = $GITHUB_REPO = 'cymetric'
$GITHUB_ORG = 'cyclus'

$ACTIVITIES = ['changelog', 'nose', 'tag', 'push_tag', 'conda_forge', 'ghrelease']
$CHANGELOG_FILENAME = 'CHANGELOG.rst'
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'

$DOCKER_CONDA_DEPS = ['cyclus', 'cymetric', 'nose', 'pytables']
$DOCKER_INSTALL_COMMAND = 'git clean -fdx && ./setup.py install --user'

$VERSION_BUMP_PATTERNS = [
    ('cymetric/__init__.py', '__version__\s*=.*', "__version__ = '$VERSION'"),
    ('setup.py', 'VERSION\s*=.*', "VERSION = '$VERSION'"),
]
