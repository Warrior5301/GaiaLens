// Type declaration for @google/earthengine
// The package ships no TypeScript types, so we declare it as `any`.
// This suppresses TS7016 without requiring @types/google__earthengine (which doesn't exist).
declare module "@google/earthengine" {
    const ee: any;
    export default ee;
    export = ee;
}
