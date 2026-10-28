//! Checks that the consumer's host file embeds each variant for exactly the GPU tiers
//! and cubin architectures that select it

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use eyre::{Context, Result};

use crate::{Project, Tier};

impl Project {
    /// The host must embed each declared variant only for tiers that select it
    pub(crate) fn host_embed_problems(&self, host_kernels: &Path) -> Result<Vec<String>> {
        let source = fs::read_to_string(host_kernels)
            .wrap_err_with(|| format!("reading {}", host_kernels.display()))?;
        Ok(self.host_embed_source_problems(&source))
    }

    fn host_embed_source_problems(&self, source: &str) -> Vec<String> {
        let host = self.host_kernels;
        let source = host_code(source);
        let embedded = embedded_ptx_files(&source);
        let declared: BTreeMap<String, Vec<String>> = self
            .areas
            .iter()
            .flat_map(|area| {
                area.variants().map(|variant| {
                    // each GPU tier uses the newest variant that this area ships
                    let features = Tier::ALL
                        .into_iter()
                        .filter(|tier| area.embeds(variant.tier, *tier))
                        .map(|tier| self.host_feature(tier))
                        .collect();
                    (variant.file_name(), features)
                })
            })
            .collect();

        let mut problems = Vec::new();
        for (name, expected) in &declared {
            let Some(uses) = embedded.get(name) else {
                problems.push(format!("{host} does not embed ptx/{name}"));
                continue;
            };

            if uses.len() != 1 {
                problems.push(format!("{host} embeds ptx/{name} more than once"));
            }

            for features in uses {
                let mut actual = features.clone().unwrap_or_default();
                actual.sort();
                let mut expected = expected.clone();
                expected.sort();
                if features.is_none() || actual != expected {
                    problems.push(format!(
                        "{host} embeds ptx/{name} with {features:?}, expected {expected:?}"
                    ));
                }
            }
        }

        // binary masks share the PTX invocation; an omitted or foreign architecture
        // would silently force JIT and invalidate production artifact matching
        for (start, _) in source.match_indices("tier_ptx!(") {
            let Some((args, _)) = source[start + "tier_ptx!(".len()..].split_once(')') else {
                continue;
            };
            let Some((_, tail)) = args.split_once(']') else {
                continue;
            };
            let Some(stem) = tail
                .split('"')
                .nth(1)
                .and_then(|path| path.strip_prefix("ptx/"))
            else {
                continue;
            };
            // old syntax occurs only in parser-negative fixtures
            if stem.ends_with(".ptx") {
                continue;
            }
            let Some(variant) = self
                .areas
                .iter()
                .flat_map(|area| area.variants())
                .find(|variant| variant.file_name() == format!("{stem}.ptx"))
            else {
                continue;
            };
            let arches = tail
                .rsplit_once('[')
                .and_then(|(_, list)| list.split_once(']'))
                .map(|(list, _)| {
                    list.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::parse::<u16>)
                        .collect::<std::result::Result<Vec<_>, _>>()
                });
            let expected: Vec<_> = self
                .variant_cubins(variant)
                .map(|cubin| cubin.arch.0)
                .collect();
            if arches != Some(Ok(expected)) {
                problems.push(format!("{host} cubin architecture mask differs for {stem}"));
            }
        }

        for name in embedded.keys().filter(|name| !declared.contains_key(*name)) {
            problems.push(format!("{host} embeds ptx/{name}, which no area declares"));
        }

        problems
    }
}

/// The host source without comment lines, so doc examples never count as embeds
///
/// Macro definitions are scanned like any other code: skipping them would need a Rust
/// tokenizer, and a consumer macro that invokes `tier_ptx!` really embeds. A wrapper
/// written as `($($variant:tt)*) => { cudakit::embed_tier!(..; $($variant)*) }` holds
/// no embed of its own, and any other wrapper fails the check loudly instead of hiding
/// real embeds
fn host_code(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .fold(String::new(), |mut code, line| {
            code.push_str(line);
            code.push('\n');
            code
        })
}

/// Every PTX inclusion, preserving duplicate calls and feature names for validation
fn embedded_ptx_files(source: &str) -> BTreeMap<String, Vec<Option<Vec<String>>>> {
    const PLAIN: &str = "include_str!(";
    const TIERED: &str = "tier_ptx!(";
    let mut files: BTreeMap<String, Vec<Option<Vec<String>>>> = BTreeMap::new();
    for (start, _) in source.match_indices(PLAIN) {
        let Some(rest) = source[start + PLAIN.len()..]
            .trim_start()
            .strip_prefix("\"ptx/")
        else {
            continue;
        };

        if let Some((name, _)) = rest.split_once('"')
            && name.ends_with(".ptx")
        {
            files.entry(name.to_string()).or_default().push(None);
        }
    }

    for (start, _) in source.match_indices(TIERED) {
        let Some((args, _)) = source[start + TIERED.len()..].split_once(')') else {
            continue;
        };

        let Some((mask, tail)) = args.split_once(']') else {
            continue;
        };
        let mask = format!("{mask}]");
        let Some(path) = tail.split('"').nth(1) else {
            continue;
        };
        let Some(stem) = path.strip_prefix("ptx/") else {
            continue;
        };
        let name = if stem.ends_with(".ptx") {
            stem.to_string()
        } else {
            format!("{stem}.ptx")
        };

        let features = mask
            .trim()
            .strip_prefix('[')
            .and_then(|mask| mask.strip_suffix(']'))
            .and_then(|mask| {
                mask.split(',')
                    .map(str::trim)
                    .filter(|feature| !feature.is_empty())
                    .map(|feature| {
                        feature
                            .strip_prefix('"')
                            .and_then(|feature| feature.strip_suffix('"'))
                            .map(str::to_string)
                    })
                    .collect()
            });
        files.entry(name).or_default().push(features);
    }

    files
}

#[cfg(test)]
mod tests {
    use crate::Tier;
    use crate::test_support::PROJECT;

    fn host_source() -> String {
        PROJECT
            .areas
            .iter()
            .flat_map(|area| area.variants())
            .map(|variant| {
                let mask = Tier::ALL
                    .into_iter()
                    .filter(|tier| variant.area.embeds(variant.tier, *tier))
                    .map(|tier| format!("{:?}", PROJECT.host_feature(tier)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("tier_ptx!([{mask}], \"ptx/{}\")", variant.file_name())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn problems(source: &str) -> Vec<String> {
        PROJECT.host_embed_source_problems(source)
    }

    #[test]
    fn host_masks_select_the_newest_shipped_variant() {
        let source = host_source();
        assert!(problems(&source).is_empty());
        assert!(source.contains("tier_ptx!([\"cuda-sm75\"], \"ptx/probe.sm75.ptx\")"));
        assert!(source.contains(
            "tier_ptx!([\"cuda-sm80\", \"cuda-sm90\", \"cuda-sm120\"], \"ptx/probe.sm80.ptx\")"
        ));
        assert!(source.contains("tier_ptx!([\"cuda-sm75\", \"cuda-sm80\", \"cuda-sm90\", \"cuda-sm120\"], \"ptx/fbank.sm75.ptx\")"));
        // a pinned baseline is embedded everywhere, and a bound top tier keeps the tier
        // below it for the same targets
        assert!(source.contains("tier_ptx!([\"cuda-sm75\", \"cuda-sm80\", \"cuda-sm90\", \"cuda-sm120\"], \"ptx/resnet.sm75.ptx\")"));
        assert!(source.contains(
            "tier_ptx!([\"cuda-sm80\", \"cuda-sm90\", \"cuda-sm120\"], \"ptx/wideconv.sm80.ptx\")"
        ));
    }

    #[test]
    fn host_masks_reject_missing_extra_and_duplicate_features() {
        for mask in [
            "[\"cuda-sm75\", \"cuda-sm80\"]",
            "[\"cuda-sm75\", \"unexpected\"]",
            "[\"cuda-sm75\", \"cuda-sm75\"]",
            "[]",
            "\"cuda-sm75\"",
        ] {
            let source = host_source().replace("[\"cuda-sm75\"]", mask);
            assert!(!problems(&source).is_empty(), "{mask}");
        }
    }

    #[test]
    fn host_inclusions_reject_unconditional_duplicate_and_unknown_files() {
        let source = host_source();
        for extra in [
            "include_str!(\"ptx/probe.sm75.ptx\")",
            "include_str!(\n \"ptx/probe.sm75.ptx\")",
            "tier_ptx!([\"cuda-sm75\"], \"ptx/probe.sm75.ptx\")",
            "tier_ptx!([\"cuda-sm75\"], \"ptx/unknown.sm75.ptx\")",
        ] {
            assert!(!problems(&format!("{source}\n{extra}")).is_empty());
        }
        assert!(!problems("").is_empty());
    }

    #[test]
    fn binary_masks_reject_missing_extra_and_foreign_architectures() {
        let source = host_source().replace(
            "\"ptx/probe.sm75.ptx\")",
            "\"ptx/probe.sm75\", [75, 80, 86, 89, 90, 120])",
        );
        assert!(problems(&source).is_empty());
        for arches in [
            "[75, 80, 86, 89, 90]",
            "[75, 80, 86, 89, 90, 120, 121]",
            "[75, 80, 86, 89, 90, 90]",
        ] {
            assert!(!problems(&source.replace("[75, 80, 86, 89, 90, 120]", arches)).is_empty());
        }
    }

    #[test]
    fn pass_through_wrappers_and_comments_are_not_embeds() {
        let wrapper = r#"
/// Embeds a variant, such as `tier_ptx!(["cuda-sm75"], "ptx/probe.sm75", [75])`
macro_rules! tier_ptx {
    ($($variant:tt)*) => {
        cudakit::embed_tier!(shipped_ptx, shipped_cubin; $($variant)*)
    };
}
// tier_ptx!(["cuda-sm75"], "ptx/unknown.sm75", [75])
"#;
        assert!(problems(&format!("{wrapper}{}", host_source())).is_empty());
    }

    #[test]
    fn embeds_inside_macro_definitions_still_count() {
        // a string brace must not hide the embeds that follow it
        let brace = "macro_rules! brace { () => { \"{\" }; }\n";
        assert!(problems(&format!("{brace}{}", host_source())).is_empty());

        // an embed in an invoked macro is real, so its masks are checked
        let invoked = "macro_rules! probe { () => { tier_ptx!([\"cuda-sm75\"], \"ptx/probe.sm80\", [80]) }; }\nprobe!();\n";
        let found = problems(&format!("{invoked}{}", host_source()));
        assert!(
            found
                .iter()
                .any(|problem| problem.contains("more than once")),
            "{found:?}"
        );
    }
}
