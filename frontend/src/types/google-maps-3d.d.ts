// Type declarations for Google Maps 3D web components
declare namespace JSX {
    interface IntrinsicElements {
        "gmp-map-3d": React.DetailedHTMLProps<
            React.HTMLAttributes<HTMLElement> & {
                center?: string;
                range?: string | number;
                tilt?: string | number;
                heading?: string | number;
                "default-ui-disabled"?: boolean | string;
            },
            HTMLElement
        >;
        "gmp-marker-3d": React.DetailedHTMLProps<
            React.HTMLAttributes<HTMLElement> & {
                position?: string;
                title?: string;
                "altitude-mode"?: string;
            },
            HTMLElement
        >;
    }
}
