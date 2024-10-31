#!/bin/bash

# Usage: ./release.sh <patch|minor|major>

set -e

if [[ -z $1 ]]; then
    echo "Please provide the version type: patch, minor, or major"
    exit 1
fi

VERSION_TYPE=$1
CURRENT_VERSION=$(grep "__version__" setup.py | cut -d '"' -f2)

# Increment version
NEW_VERSION=$(python -c "import semver; print(semver.VersionInfo.parse('$CURRENT_VERSION').bump_$VERSION_TYPE())")

# Update version in setup.py
sed -i '' "s/__version__ = \"$CURRENT_VERSION\"/__version__ = \"$NEW_VERSION\"/" setup.py

# Commit the version bump
git add setup.py
git commit -m "Release $NEW_VERSION"

# Tag the new version
git tag -a "v$NEW_VERSION" -m "Version $NEW_VERSION"
git push origin main --tags

echo "Release $NEW_VERSION created and pushed to GitHub."
