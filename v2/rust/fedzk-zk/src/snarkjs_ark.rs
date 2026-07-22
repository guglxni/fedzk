//! Convert snarkjs Groth16 JSON (BN254 / bn128) → arkworks types and verify.

use anyhow::{anyhow, bail, Context, Result};
use ark_bn254::{Bn254, Fq, Fq2, Fr, G1Affine, G2Affine};
use ark_ff::PrimeField;
use ark_groth16::{Groth16, Proof, VerifyingKey};
use ark_snark::SNARK;
use num_bigint::BigUint;
use serde_json::Value;
use std::str::FromStr;

fn fq(s: &str) -> Result<Fq> {
    let n = BigUint::from_str(s.trim()).with_context(|| format!("Fq BigUint {s}"))?;
    Ok(Fq::from_le_bytes_mod_order(&n.to_bytes_le()))
}

fn fr(s: &str) -> Result<Fr> {
    let n = BigUint::from_str(s.trim()).with_context(|| format!("Fr BigUint {s}"))?;
    Ok(Fr::from_le_bytes_mod_order(&n.to_bytes_le()))
}

fn g1_from_json(v: &Value) -> Result<G1Affine> {
    let arr = v.as_array().ok_or_else(|| anyhow!("G1 must be array"))?;
    if arr.len() < 2 {
        bail!("G1 needs at least x,y");
    }
    let x = fq(arr[0].as_str().ok_or_else(|| anyhow!("G1.x string"))?)?;
    let y = fq(arr[1].as_str().ok_or_else(|| anyhow!("G1.y string"))?)?;
    let p = G1Affine::new_unchecked(x, y);
    if !p.is_on_curve() || !p.is_in_correct_subgroup_assuming_on_curve() {
        bail!("G1 point not on curve / wrong subgroup");
    }
    Ok(p)
}

fn g2_from_json(v: &Value) -> Result<G2Affine> {
    // snarkjs: [[x_c0, x_c1], [y_c0, y_c1], [z_c0, z_c1]]
    let arr = v.as_array().ok_or_else(|| anyhow!("G2 must be array"))?;
    if arr.len() < 2 {
        bail!("G2 needs x,y Fq2");
    }
    let x_arr = arr[0]
        .as_array()
        .ok_or_else(|| anyhow!("G2.x must be [c0,c1]"))?;
    let y_arr = arr[1]
        .as_array()
        .ok_or_else(|| anyhow!("G2.y must be [c0,c1]"))?;
    let x = Fq2::new(
        fq(x_arr[0].as_str().ok_or_else(|| anyhow!("G2.x.c0"))?)?,
        fq(x_arr[1].as_str().ok_or_else(|| anyhow!("G2.x.c1"))?)?,
    );
    let y = Fq2::new(
        fq(y_arr[0].as_str().ok_or_else(|| anyhow!("G2.y.c0"))?)?,
        fq(y_arr[1].as_str().ok_or_else(|| anyhow!("G2.y.c1"))?)?,
    );
    let p = G2Affine::new_unchecked(x, y);
    if !p.is_on_curve() || !p.is_in_correct_subgroup_assuming_on_curve() {
        bail!("G2 point not on curve / wrong subgroup");
    }
    Ok(p)
}

fn vk_from_snarkjs(v: &Value) -> Result<VerifyingKey<Bn254>> {
    let alpha_g1 = g1_from_json(v.get("vk_alpha_1").context("vk_alpha_1")?)?;
    let beta_g2 = g2_from_json(v.get("vk_beta_2").context("vk_beta_2")?)?;
    let gamma_g2 = g2_from_json(v.get("vk_gamma_2").context("vk_gamma_2")?)?;
    let delta_g2 = g2_from_json(v.get("vk_delta_2").context("vk_delta_2")?)?;
    let ic_json = v
        .get("IC")
        .and_then(|x| x.as_array())
        .context("IC array")?;
    let mut gamma_abc_g1 = Vec::with_capacity(ic_json.len());
    for (i, pt) in ic_json.iter().enumerate() {
        gamma_abc_g1.push(g1_from_json(pt).with_context(|| format!("IC[{i}]"))?);
    }
    Ok(VerifyingKey {
        alpha_g1,
        beta_g2,
        gamma_g2,
        delta_g2,
        gamma_abc_g1,
    })
}

fn proof_from_snarkjs(v: &Value) -> Result<Proof<Bn254>> {
    let p = if v.get("pi_a").is_some() {
        v
    } else {
        v.get("proof").context("proof object")?
    };
    let a = g1_from_json(p.get("pi_a").context("pi_a")?)?;
    let b = g2_from_json(p.get("pi_b").context("pi_b")?)?;
    let c = g1_from_json(p.get("pi_c").context("pi_c")?)?;
    Ok(Proof { a, b, c })
}

fn publics_from_json(v: &Value) -> Result<Vec<Fr>> {
    let arr = v.as_array().context("publics array")?;
    arr.iter()
        .enumerate()
        .map(|(i, el)| {
            let s = el
                .as_str()
                .map(|x| x.to_string())
                .or_else(|| el.as_u64().map(|n| n.to_string()))
                .or_else(|| el.as_i64().map(|n| n.to_string()))
                .ok_or_else(|| anyhow!("public[{i}] not string/int"))?;
            fr(&s)
        })
        .collect()
}

pub fn verify_snarkjs(vkey: &Value, proof: &Value, public: &Value) -> Result<bool> {
    let vk = vk_from_snarkjs(vkey)?;
    let proof = proof_from_snarkjs(proof)?;
    let inputs = publics_from_json(public)?;
    if inputs.len() + 1 != vk.gamma_abc_g1.len() {
        bail!(
            "public input length mismatch: got {} publics, vk IC len {}",
            inputs.len(),
            vk.gamma_abc_g1.len()
        );
    }
    let pvk = Groth16::<Bn254>::process_vk(&vk).context("process_vk")?;
    Groth16::<Bn254>::verify_with_processed_vk(&pvk, &inputs, &proof)
        .map_err(|e| anyhow!("verify error: {e:?}"))
}
