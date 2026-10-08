import { AnalysisPage } from "@/components/shared/AnalysisPage";

export default function GeneralUsePage() {
    return (
        <AnalysisPage
            title="General Use"
            subtitle="Open-ended satellite image analysis & Q&A"
            accentColor="from-sky-500 to-blue-400"
            showTaskModes={true}
        />
    );
}
