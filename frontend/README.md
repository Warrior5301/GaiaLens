# GaiaLens frontend

The frontend is a Next.js App Router application. See the [project README](../README.md) for the architecture, environment variables, API contracts and Python setup.

## Run

From this directory, run `npm ci`, configure `.env.local` using `.env.example`, and run `npm run dev`. `npm run build` builds production output; `npm start` serves that output; `npm run lint` checks source.

## Where to edit

- `src/app/`: page routes, server API routes, root layout and global styles.
- `src/components/shared/AnalysisPage.tsx`: upload workspace and task/chat behavior.
- `src/components/ExploreEarth/ArcGISViewer.tsx`: region selection and satellite imagery interaction.
- `src/components/HomePage/`: landing page and navigation.
- `src/ui/`, `src/lib/`, `src/types/`: reusable visuals, helpers and external-library declarations.
- `next.config.ts`: the single active Next.js configuration.

Both General Use and Explore Earth use `src/lib/chat.ts` to send images and history to `/api/chat`, which proxies the FastAPI orchestrator gateway. General Use has one open-ended chat with no task-mode buttons. Explore Earth sends available observed imagery and geographic context without requiring exactly four images. Both chats display generated image attachments.

The older direct-model and Python subprocess routes remain available for compatibility but are not used by these chat components. Configure `BACKEND_API_URL` and run the gateway with its required model services.

Keep private credentials server-side. Only public browser configuration belongs in `NEXT_PUBLIC_*` variables. Local recovery copies are outside this app and ignored by Git.
