mlexpy
======

Implementation of various ML algos

Document generation
===================

There's probably a better process, but for now...

- ``git checkout master``
- ``cd docs``
- ``rm -r *``
- ``sphinx-apidoc -F -o . ../mlexpy``
- ``make html``
- Copy the ``/docs/_build/html`` directories contents elsewhere.
- ``cd ..`` (if under ``/docs`` directory)
- ``git checkout gh-pages``
- Copy contents back to root directory, replacing all but the ``.git`` folder.
- ``touch .nojekyll`` (so that GitHub renders it correctly)
- ``git add -A``
- ``git commit -m "Updating docs."``
- ``git push origin gh-pages``
- ``git checkout master``
- Continue working...
