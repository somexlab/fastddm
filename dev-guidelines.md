# Guidelines for contributions

(this is a WIP file :-) )

## Creating Issues
When you want to add something to the code, file a bug or report an issue, please create a new issue, following the provided template. Also consider adding proper labels like `bug` or `enhancement`/`task`. Please be concise in your description, adding screenshots (or mathematical formulas) may also help!
The maintainers will then add the assignee and appropriate milestone.

When working on the issue, create a new branch starting from the latest milestone branch or from `main`.

## Adding python files to installer
Python modules must be added to the `src/python` directory. To make a module installable, add it to the `src/CMakeLists.txt` file.
Scroll down until you find a list of `configure_file(...)`. Add your file to the list following the syntax of the other items:

```cmake
configure_file(python/<your_file>.py
  ${FASTDDM_OUTPUT_DIR}/<your_file>.py)
```

## Pull Requests
When opening a pull request, make sure that the source and destination branches are compatible, e.g. `<your_branch> -> <latest_milestone_branch>` or `<latest_milestone_branch> -> main`.

While you're working on the PR, please use the title format "WIP: \<title>". Once you've finished your work, remove the WIP part and ask for a review from one of the repository's maintainers.

Whenever possible, add tests for new code.

If you need advice, add comments or open a discussion, possibly tagging the appropriate maintainer.

### PR for new release
After merging the latest release into `main`, open a new WIP release PR. Do so by:
  1. create a new branch `vX.Y.Z` following the convention
  1. update the fallback version in `setup.py` and in `src/python/make_install_setup.py` and commit to the new branch
  1. open a PR with title "WIP: vX.Y.Z" with destination branch `main`.

### Merging release into `main`
1. update changelog with main additions/changes/removals/deprecations/fixes of the release you want to merge. Follow the convention in the `CHANGELOG.md`.
1. Merge the PR to main & close the corresponding milestone.
1. Create a tag via "create draft release" and name it according to the release version.
