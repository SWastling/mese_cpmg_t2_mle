# Contributing

## Releasing

1. Commit your changes:

    ```bash
    git commit -a
    ```

2. Choose a version number e.g.:

    ```bash
    version=0.0.1
    ```

3. Update the [changelog](./CHANGELOG.md)

4. Commit the updated [changelog](./CHANGELOG.md):

    ```bash
    git commit -a -m "Update changelog for $version"
    ```

5. Tag the release:

    ```bash
    git tag -a $version -m "release $version"
    ```

6. Push the changes to the GitHub repository

    ```bash
    git push -u origin main --follow-tags
    ```