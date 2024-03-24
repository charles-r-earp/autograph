# publish order

1. autograph_derive
2. autograph

# publishing

1. Create a new branch ie "publish-v0.1.0".
2. Bump all crates to the next version, removing the prerelease, ie "=0.1.0".
3. Set publish to true for workspace.
4. Commit and push the new branch.
5. PR to merge with main. Wait for CI and merge.
6. Pull the merged main.
7. Tag main with the version, ie `git tag v0.1.0`.
8. Push the tag `git push origin v0.1.0`.
9. Move into each crate directory and cargo publish.

# bumping next pre-release

1. Set publish to false for workspace.
2. Bump all versions to the next version with prerelease "alpha", ie "=0.1.1-alpha".
3. Commit and push to main.
