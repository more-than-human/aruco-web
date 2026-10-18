1. styling
   1. minimum css styling, use system default styles when possible. use system font, avoid drop shadow and gradients. feature related css file should be next to the feature related ts file. ok to use css nesting.
2. typescript
   1. prefer semantic variable and function names over verbose comments. use comments only to explain why something is done when non-obvious.
3. code organization
   1. modularize the code into cohesive files. decouple components by features so changing one thing doesn't require touching unrelated code
   2. individual files should be organized from general to specific, start with constants, then high level functions, then pure helpers in the end.
   3. `main.ts` should only contain top level logic and wiring. do not bloat it with fragmented feature code.
